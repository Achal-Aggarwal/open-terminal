"""Small exec launcher. Limits are inherited by the command and its children."""
import os
import resource
import sys


def main():
    memory, cgroup, *command = sys.argv[1:]
    if cgroup:
        with open(os.path.join(cgroup, "cgroup.procs"), "w") as stream:
            stream.write(str(os.getpid()))
    # RLIMIT_AS is enforced on Linux. Aggregate process-tree limits require a
    # delegated cgroup; per-process limits alone cannot contain arbitrary forks.
    if sys.platform == "linux" and int(memory):
        limit = int(memory)
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
