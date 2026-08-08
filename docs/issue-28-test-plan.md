# Issue #28 integration status

This foundation is intentionally not the completion of issue #28. It adds reusable
test fixtures and independent quality checks while active issues #24 and #25 own
overlapping runtime source changes.

The Unix-socket tests in this branch cover only the generic `unix_request` transport
and its fake-server fixture. A fake handler cannot validate a harness protocol. In
particular, the large-response test proves response collection is not truncated; it
does not cover oversized requests or an STT server limit.

After #24 and #25 merge, the remaining #28 integration work must add regressions
against the merged production implementations for:

- cancellation before, during, and after externally visible side effects;
- PID reuse, process identity, subprocess termination, and concurrent ownership;
- real STT request framing, including malformed and oversized requests, slow clients,
  and clients that disconnect partway through a request;
- single-winner delivery and cancellation-after-delivery boundaries;
- restart/recovery of queued, running, interrupted, and partially completed work.

Those tests should reuse `tests.support.UnixSocketServer` and
`tests.support.run_concurrently` where appropriate, then revisit the risk-specific
coverage floors. Hardware checks remain manual and are tracked separately in the
[hardware smoke checklist](hardware-smoke.md).
