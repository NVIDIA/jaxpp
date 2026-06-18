from jaxpp.utils import BoolEnvVar, IntEnvVar, StrEnvVar

jaxpp_share_donation = BoolEnvVar("JAXPP_SHARE_DONATION", True)
jaxpp_enable_local_propagation = BoolEnvVar("JAXPP_ENABLE_LOCAL_PROPAGATION", False)
jaxpp_enable_licm = BoolEnvVar("JAXPP_ENABLE_LICM", False)
jaxpp_dump_dir = StrEnvVar("JAXPP_DUMP_DIR", "")
jaxpp_disable_schedule_task_fusion = BoolEnvVar(
    "JAXPP_DISABLE_SCHEDULE_TASK_FUSION", False
)
jaxpp_enable_task_jaxpr_deduplication = BoolEnvVar(
    "JAXPP_ENABLE_TASK_JAXPR_DEDUPLICATION", True
)
jaxpp_disable_prevent_cse = BoolEnvVar("JAXPP_DISABLE_PREVENT_CSE", False)
jaxpp_directional_communicators = BoolEnvVar("JAXPP_DIRECTIONAL_COMMUNICATORS", False)
jaxpp_reuse_recv_buffers = BoolEnvVar("JAXPP_REUSE_RECV_BUFFERS", True)
jaxpp_conservative_loop_clustering = BoolEnvVar(
    "JAXPP_CONSERVATIVE_LOOP_CLUSTERING", True
)

jaxpp_debug_skip_propagation = BoolEnvVar("JAXPP_DEBUG_SKIP_PROPAGATION", False)
jaxpp_debug_force_mpmdify = BoolEnvVar("JAXPP_DEBUG_FORCE_MPMDIFY", False)

jaxpp_fast_infer_shardings = BoolEnvVar("JAXPP_FAST_INFER_SHARDINGS", False)

# When set, reconcile_shardings warns about each inferred sharding it overrides.
jaxpp_warn_reconcile_shardings = BoolEnvVar("JAXPP_WARN_RECONCILE_SHARDINGS", False)

# Timeout (ms) for blocking JAX coordination-client calls (e.g. `get_nccl_id`).
jaxpp_client_timeout = IntEnvVar("JAXPP_CLIENT_TIMEOUT", 240_000)
