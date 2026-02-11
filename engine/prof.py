import os
from datetime import datetime

def maybe_create_profiler(args):
    use_profiler = (getattr(args, "debug_profile", 0) == 2)
    if not use_profiler:
        return None
    from torch.profiler import profile, ProfilerActivity, schedule
    prof_sched = schedule(wait=5, warmup=5, active=20, repeat=1)
    p = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        on_trace_ready=lambda prof: prof.export_chrome_trace(
            os.path.join(args.log_dir, f"trace_rank0_{datetime.now().strftime('%H%M%S')}.json")
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )
    p.start()
    return p