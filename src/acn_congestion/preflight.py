"""Read-only testbed preflight; never creates namespaces, interfaces, routes, or qdiscs."""

from __future__ import annotations

import ipaddress
import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from .parsing import parse_backlog_bytes, parse_duration_ms, parse_rate_bps


class PreflightError(ValueError):
    """A testbed preflight could not complete its read-only checks."""


def _run(
    command: Sequence[str],
    *,
    timeout: float = 5.0,
    runner: Callable[..., Any] = subprocess.run,
    privileged: bool = False,
) -> dict[str, Any]:
    """Run a read-only inspection command, optionally through non-interactive sudo."""
    actual_command = list(command)

    if privileged:
        actual_command = ["sudo", "-n", *actual_command]

    try:
        result = runner(
            actual_command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": actual_command,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "command": actual_command,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def preflight(
    config: dict[str, Any],
    *,
    runner: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    """Inspect tools, namespaces, addresses, routes, forwarding and qdisc read-only."""
    topology = config["topology"]

    expected_namespaces = [
        topology[key]
        for key in (
            "client_namespace",
            "router_namespace",
            "server_namespace",
        )
    ]

    required_tools = ("ip", "tc", "ping", "iperf3")
    tools = {name: which(name) for name in required_tools}

    checks: list[dict[str, Any]] = [
        {
            "name": f"tool:{name}",
            "ok": path is not None,
            "path": path,
            "detail": (
                "present"
                if path
                else f"required executable {name!r} is missing"
            ),
        }
        for name, path in tools.items()
    ]

    if not tools["ip"]:
        return {
            "status": "blocked",
            "read_only": True,
            "connectivity_tested": False,
            "tools": tools,
            "checks": checks,
            "note": "The ip executable is unavailable. No setup or network changes were attempted.",
        }

    # Namespace enumeration is permitted without elevated privileges.
    namespace_result = _run(
        [tools["ip"], "netns", "list"],
        runner=runner,
    )
    checks.append({"name": "namespace_list", **namespace_result})

    if not namespace_result["ok"]:
        return {
            "status": "blocked",
            "read_only": True,
            "connectivity_tested": False,
            "tools": tools,
            "checks": checks,
            "note": "Namespace enumeration failed; no setup was attempted.",
        }

    present = {
        line.split()[0]
        for line in namespace_result["stdout"].splitlines()
        if line.split()
    }

    missing = sorted(set(expected_namespaces) - present)

    checks.append(
        {
            "name": "expected_namespaces",
            "ok": not missing,
            "expected": expected_namespaces,
            "present": sorted(present),
            "missing": missing,
        }
    )

    if missing:
        return {
            "status": "not_ready",
            "read_only": True,
            "connectivity_tested": False,
            "tools": tools,
            "checks": checks,
            "note": "Missing namespaces are not created automatically. No changes were attempted.",
        }

    # These addresses and interface names come from experiment.yaml.
    interface_expectations = [
        (
            topology["client_namespace"],
            topology["client_interface"],
            topology["client_address"],
        ),
        (
            topology["router_namespace"],
            topology["router_client_interface"],
            topology["router_client_address"],
        ),
        (
            topology["router_namespace"],
            topology["router_server_interface"],
            topology["router_server_address"],
        ),
        (
            topology["server_namespace"],
            topology["server_interface"],
            topology["server_address"],
        ),
    ]

    # Inspect addresses using sudo -n because entering namespaces requires privileges.
    for namespace, interface, expected_cidr in interface_expectations:
        command = [
            tools["ip"],
            "-n",
            namespace,
            "-j",
            "address",
            "show",
            "dev",
            interface,
        ]

        result = _run(
            command,
            runner=runner,
            privileged=True,
        )

        check: dict[str, Any] = {
            "name": f"address:{namespace}:{interface}",
            "ok": False,
            "expected_cidr": expected_cidr,
            "command_result": result,
        }

        if result["ok"]:
            try:
                addresses = json.loads(result["stdout"])

                observed = [
                    f'{info["local"]}/{info["prefixlen"]}'
                    for link in addresses
                    for info in link.get("addr_info", [])
                    if info.get("family") == "inet"
                    and info.get("local") is not None
                    and info.get("prefixlen") is not None
                ]

                check["observed_cidrs"] = observed
                check["ok"] = expected_cidr in observed

            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                check["parse_error"] = str(exc)

        checks.append(check)

    # Derive expected routes from the configured IP addresses.
    try:
        client_network = str(
            ipaddress.ip_interface(topology["client_address"]).network
        )
        server_network = str(
            ipaddress.ip_interface(topology["server_address"]).network
        )
        client_gateway = str(
            ipaddress.ip_interface(topology["router_client_address"]).ip
        )
        server_gateway = str(
            ipaddress.ip_interface(topology["router_server_address"]).ip
        )

    except (KeyError, ValueError) as exc:
        raise PreflightError(
            f"invalid topology addressing in configuration: {exc}"
        ) from exc

    expected_routes = [
        (
            topology["client_namespace"],
            server_network,
            client_gateway,
        ),
        (
            topology["server_namespace"],
            client_network,
            server_gateway,
        ),
    ]

    for namespace, destination, gateway in expected_routes:
        result = _run(
            [
                tools["ip"],
                "-n",
                namespace,
                "-j",
                "route",
                "show",
            ],
            runner=runner,
            privileged=True,
        )

        check: dict[str, Any] = {
            "name": f"route:{namespace}:{destination}",
            "ok": False,
            "expected_gateway": gateway,
            "command_result": result,
        }

        if result["ok"]:
            try:
                routes = json.loads(result["stdout"])

                matches = [
                    route
                    for route in routes
                    if route.get("dst") == destination
                    and route.get("gateway") == gateway
                ]

                check["observed_routes"] = routes
                check["ok"] = bool(matches)

            except (json.JSONDecodeError, TypeError) as exc:
                check["parse_error"] = str(exc)

        checks.append(check)

    router = topology["router_namespace"]

    # Read the router forwarding setting without changing it.
    forwarding = _run(
        [
            tools["ip"],
            "netns",
            "exec",
            router,
            "cat",
            "/proc/sys/net/ipv4/ip_forward",
        ],
        runner=runner,
        privileged=True,
    )

    checks.append(
        {
            "name": "router_ipv4_forwarding",
            "ok": forwarding["ok"] and forwarding.get("stdout") == "1",
            "observed": forwarding.get("stdout"),
            "command_result": forwarding,
        }
    )

    egress = topology["bottleneck_egress"]

    # Inspect the bottleneck qdisc without modifying it.
    if tools["tc"]:
        qdisc = _run(
            [
                tools["ip"],
                "netns",
                "exec",
                router,
                tools["tc"],
                "-j",
                "-s",
                "qdisc",
                "show",
                "dev",
                egress,
            ],
            runner=runner,
            privileged=True,
        )

        qdisc_check: dict[str, Any] = {
            "name": "bottleneck_qdisc",
            "ok": False,
            "command_result": qdisc,
        }

        if qdisc["ok"]:
            try:
                parsed = json.loads(qdisc["stdout"])
                tbf = [
                    item
                    for item in parsed
                    if item.get("kind") == "tbf"
                ]

                qdisc_check["observed_qdiscs"] = parsed
                qdisc_check["expected_rate_bps"] = topology.get(
                    "bottleneck_rate_bps"
                )

                if len(tbf) == 1:
                    options = tbf[0].get("options", {})

                    observed_rate = options.get(
                        "rate64",
                        options.get("rate"),
                    )
                    qdisc_check["observed_rate"] = observed_rate

                    try:
                        # Linux tc JSON reports numeric TBF rate in bytes/second.
                        if isinstance(observed_rate, (int, float)) and not isinstance(observed_rate, bool):
                            observed_rate_bps = int(round(observed_rate * 8))
                        else:
                            observed_rate_bps = parse_rate_bps(observed_rate)

                        qdisc_check["observed_rate_bps"] = observed_rate_bps
                    except (ValueError, TypeError) as exc:
                        qdisc_check["rate_parse_error"] = str(exc)
                        observed_rate_bps = None

                    expected_burst = topology.get(
                        "bottleneck_burst_bytes"
                    )
                    expected_latency = topology.get(
                        "bottleneck_latency_ms"
                    )

                    if expected_burst is not None:
                        try:
                            observed_burst = parse_backlog_bytes(
                                options.get("burst")
                            )
                            qdisc_check["observed_burst_bytes"] = observed_burst
                            qdisc_check["expected_burst_bytes"] = expected_burst
                            burst_ok = observed_burst == expected_burst

                        except ValueError as exc:
                            qdisc_check["burst_parse_error"] = str(exc)
                            burst_ok = False
                    else:
                        burst_ok = True

                    if expected_latency is not None:
                        try:
                            raw_latency = options.get(
                                "latency",
                                options.get("lat"),
                            )

                            # Linux tc JSON reports numeric latency in microseconds.
                            if isinstance(raw_latency, (int, float)) and not isinstance(raw_latency, bool):
                                observed_latency = raw_latency / 1000.0
                            else:
                                observed_latency = parse_duration_ms(raw_latency)

                            qdisc_check["observed_latency_ms"] = observed_latency
                            qdisc_check["expected_latency_ms"] = expected_latency
                            latency_ok = observed_latency == expected_latency

                        except ValueError as exc:
                            qdisc_check["latency_parse_error"] = str(exc)
                            latency_ok = False
                    else:
                        latency_ok = True

                    qdisc_check["ok"] = (
                        observed_rate_bps
                        == topology.get("bottleneck_rate_bps")
                        and burst_ok
                        and latency_ok
                    )

            except (json.JSONDecodeError, TypeError) as exc:
                qdisc_check["parse_error"] = str(exc)

        checks.append(qdisc_check)

    else:
        checks.append(
            {
                "name": "bottleneck_qdisc",
                "ok": False,
                "detail": "tc executable unavailable; qdisc was not inspected",
            }
        )

    ready = all(check.get("ok", False) for check in checks)

    return {
        "status": "ready_for_manual_review" if ready else "not_ready",
        "read_only": True,
        "tools": tools,
        "checks": checks,
        "connectivity_tested": False,
        "note": (
            "Read-only topology inspection only. No packet probes, traffic "
            "generation, or network changes were performed. A ready status "
            "is not a successful experiment."
        ),
    }
