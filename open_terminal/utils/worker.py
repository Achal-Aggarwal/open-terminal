"""Exec launcher for a supervisor-provided workload cgroup."""
import os
import sys


def main():
    cgroup, *command = sys.argv[1:]
    with open(os.path.join(cgroup, "cgroup.procs"), "w") as stream:
        stream.write(str(os.getpid()))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
