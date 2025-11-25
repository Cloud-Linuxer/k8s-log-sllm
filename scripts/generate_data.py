#!/usr/bin/env python3
"""Generate realistic synthetic log data for training."""

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LogGenerator:
    """Generate realistic synthetic log data."""

    # K8s components
    K8S_COMPONENTS = [
        "kubelet", "kube-apiserver", "kube-scheduler", "kube-controller-manager",
        "etcd", "kube-proxy", "coredns", "calico-node", "flannel", "containerd",
    ]

    # Application names
    APP_NAMES = [
        "nginx", "redis", "postgres", "mysql", "mongodb", "elasticsearch",
        "kafka", "rabbitmq", "prometheus", "grafana", "jaeger", "istio-proxy",
        "envoy", "vault", "consul", "traefik", "haproxy", "memcached",
    ]

    # System services
    SYSTEM_SERVICES = [
        "systemd", "sshd", "docker", "containerd", "cron", "rsyslog",
        "auditd", "NetworkManager", "firewalld", "chronyd", "dbus",
    ]

    # Namespaces
    NAMESPACES = [
        "default", "kube-system", "monitoring", "logging", "ingress-nginx",
        "cert-manager", "istio-system", "production", "staging", "development",
    ]

    # Pod name prefixes
    POD_PREFIXES = [
        "web", "api", "worker", "scheduler", "gateway", "auth", "user",
        "order", "payment", "notification", "search", "cache", "queue",
    ]

    # Node names
    NODE_NAMES = [
        "node-1", "node-2", "node-3", "worker-01", "worker-02", "worker-03",
        "master-01", "master-02", "master-03", "compute-a1", "compute-b2",
    ]

    # HTTP endpoints
    HTTP_ENDPOINTS = [
        "/api/v1/users", "/api/v1/orders", "/api/v1/products", "/api/v1/auth/login",
        "/api/v1/auth/logout", "/api/v2/health", "/api/v2/metrics", "/healthz",
        "/readyz", "/livez", "/api/v1/webhook", "/api/internal/sync",
    ]

    # Error messages by category
    K8S_ERRORS = {
        "critical": [
            "CrashLoopBackOff",
            "ImagePullBackOff",
            "Back-off restarting failed container",
            "Failed to create pod sandbox",
            "failed to create containerd task",
            "OOMKilled",
            "Evicted",
            "NodeNotReady",
            "FailedScheduling",
            "FailedMount",
        ],
        "warning": [
            "Liveness probe failed",
            "Readiness probe failed",
            "Container creating",
            "Pulling image",
            "Failed to pull image",
            "context deadline exceeded",
            "Backoff pulling image",
            "FailedGetResourceMetric",
            "ProbeWarning",
        ],
        "info": [
            "Started container",
            "Created container",
            "Pulled image",
            "Successfully assigned",
            "Container started",
            "Volume mounted",
            "ConfigMap mounted",
            "Secret mounted",
        ],
    }

    SYSTEM_ERRORS = {
        "critical": [
            "kernel panic - not syncing",
            "BUG: unable to handle kernel paging request",
            "segfault at",
            "Out of memory: Kill process",
            "I/O error, dev",
            "EXT4-fs error",
            "XFS: Internal error",
            "Hardware Error",
            "MCE: CPU thermal throttling",
        ],
        "warning": [
            "Connection timed out",
            "Connection refused",
            "Connection reset by peer",
            "No route to host",
            "disk usage above threshold",
            "memory usage high",
            "load average exceeded",
            "swap usage detected",
        ],
        "info": [
            "Started",
            "Stopped",
            "Reloaded",
            "Listening on",
            "Accepted publickey",
            "session opened",
            "session closed",
            "Startup finished",
        ],
    }

    NETWORK_ERRORS = {
        "critical": [
            "upstream timed out (110: Connection timed out)",
            "no live upstreams",
            "connect() failed (111: Connection refused)",
            "SSL_do_handshake() failed",
            "certificate has expired",
            "peer closed connection in SSL handshake",
        ],
        "warning": [
            "upstream prematurely closed connection",
            "client intended to send too large body",
            "limiting requests",
            "limiting connections",
            "recv() failed (104: Connection reset by peer)",
        ],
        "info": [
            "GET /health 200",
            "POST /api/v1 200",
            "GET /metrics 200",
            "HEAD /healthz 200",
        ],
    }

    SECURITY_ERRORS = {
        "critical": [
            "authentication failed",
            "POSSIBLE BREAK-IN ATTEMPT",
            "Invalid user",
            "Failed password for invalid user",
            "maximum authentication attempts exceeded",
        ],
        "warning": [
            "Permission denied",
            "Unauthorized",
            "Forbidden",
            "access denied",
            "Invalid token",
            "Token expired",
            "rate limit exceeded",
        ],
        "info": [
            "Accepted password for",
            "Accepted publickey for",
            "session opened for user",
            "New session",
        ],
    }

    APP_ERRORS = {
        "critical": [
            "FATAL: terminating connection due to administrator command",
            "FATAL: too many connections",
            "FATAL: password authentication failed",
            "PANIC: could not locate",
            "deadlock detected",
            "out of memory",
            "stack overflow",
            "unhandled exception",
        ],
        "warning": [
            "slow query detected",
            "connection pool exhausted",
            "cache miss rate high",
            "queue backlog detected",
            "retry attempt",
            "circuit breaker open",
            "timeout waiting for",
        ],
        "info": [
            "query executed successfully",
            "connection established",
            "cache hit",
            "request completed",
            "transaction committed",
            "index created",
            "backup completed",
        ],
    }

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        random.seed(seed)
        self.base_time = datetime.now() - timedelta(days=7)

    def _gen_timestamp(self) -> str:
        """Generate ISO8601 timestamp."""
        offset = random.randint(0, 7 * 24 * 3600)
        ts = self.base_time + timedelta(seconds=offset)
        return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{random.randint(0, 999):03d}Z"

    def _gen_syslog_timestamp(self) -> str:
        """Generate syslog timestamp."""
        offset = random.randint(0, 7 * 24 * 3600)
        ts = self.base_time + timedelta(seconds=offset)
        return ts.strftime("%b %d %H:%M:%S")

    def _gen_ip(self) -> str:
        """Generate random IP address."""
        ip_type = random.choice(["internal", "internal", "internal", "external"])
        if ip_type == "internal":
            prefix = random.choice(["10.0", "10.1", "10.244", "172.17", "192.168.1"])
            return f"{prefix}.{random.randint(1, 254)}.{random.randint(1, 254)}"
        else:
            return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

    def _gen_pod_name(self) -> str:
        """Generate Kubernetes pod name."""
        prefix = random.choice(self.POD_PREFIXES)
        deployment_hash = ''.join(random.choices('abcdef0123456789', k=10))
        pod_hash = ''.join(random.choices('abcdef0123456789', k=5))
        return f"{prefix}-{deployment_hash}-{pod_hash}"

    def _gen_container_id(self) -> str:
        """Generate container ID."""
        return ''.join(random.choices('abcdef0123456789', k=12))

    def _gen_uuid(self) -> str:
        """Generate UUID."""
        return str(uuid.uuid4())

    def _gen_pid(self) -> int:
        """Generate PID."""
        return random.randint(1, 65535)

    def _gen_port(self) -> int:
        """Generate port number."""
        return random.choice([80, 443, 8080, 8443, 3000, 5000, 6379, 5432, 3306, 9090, 9200])

    def _gen_latency(self) -> str:
        """Generate latency value."""
        if random.random() < 0.8:
            return f"{random.randint(1, 100)}ms"
        elif random.random() < 0.95:
            return f"{random.randint(100, 1000)}ms"
        else:
            return f"{random.randint(1, 30)}s"

    def _gen_http_status(self, error_level: str) -> int:
        """Generate HTTP status code based on error level."""
        if error_level == "critical":
            return random.choice([500, 502, 503, 504])
        elif error_level == "warning":
            return random.choice([400, 401, 403, 404, 408, 429])
        else:
            return random.choice([200, 201, 204, 301, 302])

    def generate_k8s_log(self, level: str = None) -> str:
        """Generate Kubernetes log."""
        if level is None:
            level = random.choices(
                ["info", "warning", "critical"],
                weights=[0.7, 0.2, 0.1]
            )[0]

        component = random.choice(self.K8S_COMPONENTS)
        namespace = random.choice(self.NAMESPACES)
        pod = self._gen_pod_name()
        node = random.choice(self.NODE_NAMES)
        ts = self._gen_timestamp()

        level_map = {"info": "INFO", "warning": "WARNING", "critical": "ERROR"}
        log_level = level_map[level]

        if level == "critical":
            error = random.choice(self.K8S_ERRORS["critical"])
            templates = [
                f"{ts} {log_level} {component}[{self._gen_pid()}]: {error} for pod {namespace}/{pod} on node {node}",
                f"{ts} {log_level} {component}: Pod {namespace}/{pod} {error}",
                f"{ts} {log_level} {component}: container {self._gen_container_id()} {error}",
                f"{ts} {log_level} {component}[{self._gen_pid()}]: {error}: context deadline exceeded after 30s",
                f"{ts} {log_level} {component}: {error} - pod {pod} in namespace {namespace}",
            ]
        elif level == "warning":
            error = random.choice(self.K8S_ERRORS["warning"])
            templates = [
                f"{ts} {log_level} {component}[{self._gen_pid()}]: {error} for container in pod {namespace}/{pod}",
                f"{ts} {log_level} {component}: {error} - retrying in {random.randint(5, 60)}s",
                f"{ts} {log_level} {component}: {error} from {self._gen_ip()}:{self._gen_port()}",
                f"{ts} {log_level} {component}[{self._gen_pid()}]: {error} for pod {pod}",
            ]
        else:
            msg = random.choice(self.K8S_ERRORS["info"])
            templates = [
                f"{ts} {log_level} {component}[{self._gen_pid()}]: {msg} {pod} in namespace {namespace}",
                f"{ts} {log_level} {component}: {msg} for pod {namespace}/{pod}",
                f"{ts} {log_level} {component}: Successfully {msg.lower()} on node {node}",
                f"{ts} {log_level} {component}[{self._gen_pid()}]: {msg}",
            ]

        return random.choice(templates)

    def generate_system_log(self, level: str = None) -> str:
        """Generate system log."""
        if level is None:
            level = random.choices(
                ["info", "warning", "critical"],
                weights=[0.75, 0.18, 0.07]
            )[0]

        service = random.choice(self.SYSTEM_SERVICES)
        hostname = random.choice(["server1", "server2", "worker1", "master1", "node1"])
        ts = self._gen_syslog_timestamp()
        pid = self._gen_pid()

        if level == "critical":
            error = random.choice(self.SYSTEM_ERRORS["critical"])
            templates = [
                f"{ts} {hostname} kernel: {error} at {hex(random.randint(0, 0xFFFFFFFF))}",
                f"{ts} {hostname} {service}[{pid}]: CRITICAL: {error}",
                f"{ts} {hostname} kernel: BUG: {error}",
                f"{ts} {hostname} {service}[{pid}]: FATAL: {error}",
            ]
        elif level == "warning":
            error = random.choice(self.SYSTEM_ERRORS["warning"])
            templates = [
                f"{ts} {hostname} {service}[{pid}]: WARNING: {error}",
                f"{ts} {hostname} {service}[{pid}]: {error} - {self._gen_ip()}",
                f"{ts} {hostname} kernel: {error} on device sda{random.randint(1, 5)}",
            ]
        else:
            msg = random.choice(self.SYSTEM_ERRORS["info"])
            templates = [
                f"{ts} {hostname} {service}[{pid}]: {msg}",
                f"{ts} {hostname} systemd[1]: {msg} {service}.service",
                f"{ts} {hostname} {service}[{pid}]: INFO: {msg}",
            ]

        return random.choice(templates)

    def generate_nginx_log(self, level: str = None) -> str:
        """Generate nginx/proxy log."""
        if level is None:
            level = random.choices(
                ["info", "warning", "critical"],
                weights=[0.8, 0.15, 0.05]
            )[0]

        ts = self._gen_timestamp()
        client_ip = self._gen_ip()
        upstream_ip = self._gen_ip()
        endpoint = random.choice(self.HTTP_ENDPOINTS)
        method = random.choice(["GET", "POST", "PUT", "DELETE", "PATCH"])
        status = self._gen_http_status(level)
        latency = self._gen_latency()

        if level == "critical":
            error = random.choice(self.NETWORK_ERRORS["critical"])
            templates = [
                f"{ts} ERROR nginx: {error} while reading response header from upstream, client: {client_ip}, upstream: {upstream_ip}:{self._gen_port()}",
                f"{ts} ERROR nginx: {error}, server: _, request: \"{method} {endpoint}\"",
                f"{ts} CRITICAL nginx[{self._gen_pid()}]: {error} - upstream: {upstream_ip}",
            ]
        elif level == "warning":
            error = random.choice(self.NETWORK_ERRORS["warning"])
            templates = [
                f"{ts} WARNING nginx: {error}, client: {client_ip}, request: \"{method} {endpoint}\"",
                f"{ts} WARN nginx[{self._gen_pid()}]: {error}",
            ]
        else:
            templates = [
                f"{ts} INFO nginx: {client_ip} - - \"{method} {endpoint} HTTP/1.1\" {status} {random.randint(100, 10000)} \"-\" \"Mozilla/5.0\" {latency}",
                f"{ts} INFO nginx[{self._gen_pid()}]: {method} {endpoint} {status} {latency}",
            ]

        return random.choice(templates)

    def generate_app_log(self, level: str = None) -> str:
        """Generate application log."""
        if level is None:
            level = random.choices(
                ["info", "warning", "critical"],
                weights=[0.75, 0.18, 0.07]
            )[0]

        app = random.choice(self.APP_NAMES)
        ts = self._gen_timestamp()
        request_id = self._gen_uuid()

        level_map = {"info": "INFO", "warning": "WARN", "critical": "ERROR"}
        log_level = level_map[level]

        if level == "critical":
            error = random.choice(self.APP_ERRORS["critical"])
            templates = [
                f"{ts} {log_level} [{app}] [request_id={request_id}] {error}",
                f"{ts} {log_level} {app}[{self._gen_pid()}]: {error}",
                f"{ts} FATAL [{app}] {error} - shutting down",
                f"{ts} {log_level} [{app}] {error} at {self._gen_ip()}:{self._gen_port()}",
            ]
        elif level == "warning":
            error = random.choice(self.APP_ERRORS["warning"])
            templates = [
                f"{ts} {log_level} [{app}] [request_id={request_id}] {error}",
                f"{ts} {log_level} {app}: {error} - latency: {self._gen_latency()}",
                f"{ts} {log_level} [{app}] {error}",
            ]
        else:
            msg = random.choice(self.APP_ERRORS["info"])
            templates = [
                f"{ts} {log_level} [{app}] [request_id={request_id}] {msg}",
                f"{ts} {log_level} {app}[{self._gen_pid()}]: {msg}",
                f"{ts} {log_level} [{app}] {msg} - latency: {self._gen_latency()}",
            ]

        return random.choice(templates)

    def generate_security_log(self, level: str = None) -> str:
        """Generate security-related log."""
        if level is None:
            level = random.choices(
                ["info", "warning", "critical"],
                weights=[0.6, 0.25, 0.15]
            )[0]

        ts = self._gen_syslog_timestamp()
        hostname = random.choice(["server1", "server2", "bastion", "gateway"])
        client_ip = self._gen_ip()
        user = random.choice(["root", "admin", "user", "deploy", "operator", "guest"])

        if level == "critical":
            error = random.choice(self.SECURITY_ERRORS["critical"])
            templates = [
                f"{ts} {hostname} sshd[{self._gen_pid()}]: {error} {user} from {client_ip} port {random.randint(10000, 65535)}",
                f"{ts} {hostname} auth[{self._gen_pid()}]: CRITICAL: {error} for user {user}",
                f"{ts} {hostname} sshd[{self._gen_pid()}]: error: {error}",
            ]
        elif level == "warning":
            error = random.choice(self.SECURITY_ERRORS["warning"])
            templates = [
                f"{ts} {hostname} sshd[{self._gen_pid()}]: {error} for {user} from {client_ip}",
                f"{ts} {hostname} auth[{self._gen_pid()}]: WARNING: {error}",
                f"{ts} {hostname} audit[{self._gen_pid()}]: {error} - user={user} ip={client_ip}",
            ]
        else:
            msg = random.choice(self.SECURITY_ERRORS["info"])
            templates = [
                f"{ts} {hostname} sshd[{self._gen_pid()}]: {msg} {user} from {client_ip}",
                f"{ts} {hostname} auth[{self._gen_pid()}]: INFO: {msg}",
                f"{ts} {hostname} systemd-logind[{self._gen_pid()}]: {msg} {user}",
            ]

        return random.choice(templates)

    def generate_database_log(self, level: str = None) -> str:
        """Generate database log."""
        if level is None:
            level = random.choices(
                ["info", "warning", "critical"],
                weights=[0.7, 0.2, 0.1]
            )[0]

        db = random.choice(["postgres", "mysql", "mongodb", "redis"])
        ts = self._gen_timestamp()
        client_ip = self._gen_ip()

        level_map = {"info": "LOG", "warning": "WARNING", "critical": "ERROR"}
        log_level = level_map[level]

        if level == "critical":
            errors = [
                "FATAL: too many connections for role",
                "FATAL: password authentication failed",
                "ERROR: deadlock detected",
                "FATAL: could not open relation",
                "ERROR: duplicate key value violates unique constraint",
                "PANIC: could not write to file",
                "ERROR: out of memory",
            ]
            error = random.choice(errors)
            templates = [
                f"{ts} {log_level} {db}[{self._gen_pid()}]: {error}",
                f"{ts} {log_level} [{db}] {error} for user \"app\" from {client_ip}",
            ]
        elif level == "warning":
            warnings = [
                "checkpoints are occurring too frequently",
                "connection pool exhausted",
                "slow query",
                "table scan detected",
                "high memory usage",
                "replication lag detected",
            ]
            warn = random.choice(warnings)
            templates = [
                f"{ts} {log_level} {db}[{self._gen_pid()}]: {warn}",
                f"{ts} {log_level} [{db}] {warn} - duration: {self._gen_latency()}",
            ]
        else:
            msgs = [
                "connection authorized",
                "statement executed",
                "checkpoint complete",
                "autovacuum completed",
                "database system is ready",
                "received fast shutdown request",
            ]
            msg = random.choice(msgs)
            templates = [
                f"{ts} {log_level} {db}[{self._gen_pid()}]: {msg}",
                f"{ts} {log_level} [{db}] {msg}",
            ]

        return random.choice(templates)

    def generate_debug_log(self, level: str = None) -> str:
        """Generate debug/trace log."""
        app = random.choice(self.APP_NAMES + self.K8S_COMPONENTS)
        ts = self._gen_timestamp()

        debug_msgs = [
            "Entering function",
            "Exiting function",
            "Variable state",
            "Checkpoint reached",
            "Processing item",
            "Loop iteration",
            "Cache lookup",
            "Memory allocation",
            "Buffer flush",
            "Timer tick",
            "Event received",
            "State transition",
            "Context switch",
            "Lock acquired",
            "Lock released",
        ]

        trace_msgs = [
            "TRACE: method entry",
            "TRACE: method exit",
            "TRACE: parameter value",
            "TRACE: return value",
            "TRACE: stack frame",
            "TRACE: heap status",
            "TRACE: gc cycle",
        ]

        if level == "trace" or (level is None and random.random() < 0.3):
            msg = random.choice(trace_msgs)
            log_level = "TRACE"
        else:
            msg = random.choice(debug_msgs)
            log_level = "DEBUG"

        templates = [
            f"{ts} {log_level} [{app}] {msg} - id={random.randint(1, 1000)}",
            f"{ts} {log_level} {app}[{self._gen_pid()}]: {msg}",
            f"{ts} {log_level} [{app}] {msg}: value={random.randint(0, 100)}",
            f"{ts} {log_level} {app}: {msg} at line {random.randint(1, 500)}",
        ]

        return random.choice(templates)

    def generate_log(self, target_level: str = None) -> str:
        """Generate a random log entry with optional target level."""
        if target_level in ["trace", "debug"]:
            return self.generate_debug_log(target_level)

        generator = random.choices(
            [
                self.generate_k8s_log,
                self.generate_system_log,
                self.generate_nginx_log,
                self.generate_app_log,
                self.generate_security_log,
                self.generate_database_log,
            ],
            weights=[0.25, 0.15, 0.2, 0.2, 0.1, 0.1]
        )[0]

        if target_level:
            return generator(level=target_level)
        return generator()

    def generate_batch(self, count: int, balanced: bool = False) -> list[str]:
        """Generate batch of logs.

        Args:
            count: Number of logs to generate
            balanced: If True, generate balanced distribution across severity levels
        """
        if not balanced:
            return [self.generate_log() for _ in range(count)]

        # Balanced distribution across severity levels
        # Target distribution: trace/debug (10%), info (35%), warning (25%), error (20%), critical (10%)
        logs = []

        # Calculate counts for each level
        trace_debug_count = int(count * 0.10)
        info_count = int(count * 0.35)
        warning_count = int(count * 0.25)
        error_count = int(count * 0.20)
        critical_count = count - trace_debug_count - info_count - warning_count - error_count

        # Generate logs for each level
        for _ in range(trace_debug_count):
            logs.append(self.generate_debug_log())

        for _ in range(info_count):
            logs.append(self.generate_log(target_level="info"))

        for _ in range(warning_count):
            logs.append(self.generate_log(target_level="warning"))

        for _ in range(error_count):
            generator = random.choice([
                self.generate_k8s_log,
                self.generate_system_log,
                self.generate_nginx_log,
                self.generate_app_log,
                self.generate_security_log,
                self.generate_database_log,
            ])
            # Generate error-level logs (will become label 6-7)
            logs.append(generator(level="warning"))  # Some warnings have keywords that push them up

        for _ in range(critical_count):
            logs.append(self.generate_log(target_level="critical"))

        # Shuffle to mix levels
        random.shuffle(logs)
        return logs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic log data")
    parser.add_argument(
        "--count", "-n", type=int, default=10000, help="Number of logs to generate (default: 10000)"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output file path"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--with-labels", action="store_true", help="Also run preprocessing and labeling"
    )
    parser.add_argument(
        "--balanced", "-b", action="store_true", help="Generate balanced distribution across severity levels"
    )

    args = parser.parse_args()

    print(f"Generating {args.count} synthetic logs{'(balanced)' if args.balanced else ''}...")
    generator = LogGenerator(seed=args.seed)
    logs = generator.generate_batch(args.count, balanced=args.balanced)

    # Save raw logs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(log + "\n")

    print(f"Saved {len(logs)} logs to {args.output}")

    # Optionally preprocess and label
    if args.with_labels:
        from src.preprocessing import LogPreprocessor
        from src.labeling import RiskLabeler
        import pandas as pd

        print("Preprocessing logs...")
        preprocessor = LogPreprocessor()
        processed = preprocessor.preprocess_batch(logs)

        print("Generating labels...")
        labeler = RiskLabeler()
        labels = labeler.label_batch(processed)

        # Create DataFrame
        df = pd.DataFrame(processed)
        df["original"] = logs
        df["label"] = labels

        # Save labeled data
        labeled_path = output_path.with_suffix(".labeled.csv")
        df.to_csv(labeled_path, index=False)
        print(f"Saved labeled data to {labeled_path}")

        # Print distribution
        print("\nLabel distribution:")
        for i in range(11):
            count = sum(1 for l in labels if l == i)
            pct = count / len(labels) * 100
            bar = "#" * int(pct / 2)
            print(f"  {i:2d}: {count:5d} ({pct:5.1f}%) {bar}")

        # Print level distribution
        print("\nLog level distribution:")
        levels = [p["level_raw"] or "NONE" for p in processed]
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        for level, count in sorted(level_counts.items(), key=lambda x: -x[1]):
            print(f"  {level}: {count}")


if __name__ == "__main__":
    main()
