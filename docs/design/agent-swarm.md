# The tensor4all agent swarm

Status: proposal (2026-11), not built yet.

The design of the agent swarm moved to
[shinaoka/agent-swarm](https://github.com/shinaoka/agent-swarm) (private for
now; ask the owner for access). It grew to run several projects (tensor4all,
Koushi, sparse-ir) on shared machines, so it no longer belongs to tensor4all
alone.

What it means for tensor4all:

- tensor4all is one project on the swarm, with the room prefix `t4a`. You give
  instructions to the tensor4all chief in `#t4a-chief`; the other `#t4a-*`
  rooms are for reading.
- tensor4all keeps its own superagents, memory, GitHub access, LLM keys and
  operators. Other projects on the swarm cannot see or touch them.
- The rules tensor4all agents follow stay in
  [tensor4all-agent-rules](https://github.com/tensor4all/tensor4all-agent-rules),
  and the fleet described there stays authoritative until tensor4all's cutover
  day.
