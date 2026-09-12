from ai_fingerprint.packet_fingerprinting import PacketObservation, iter_packet_budget_samples, normalize_direction


def p(t, n, d):
    return PacketObservation(t, n, d, "TCP", 0, 1, 0, 0, 0, ())

packets = [p(0.0, 100, "up"), p(0.1, 200, "down"), p(0.2, 300, "up"), p(0.3, 400, "up")]
assert len(list(iter_packet_budget_samples(packets, 1))) == 4
blocks = list(iter_packet_budget_samples(packets, 2))
assert len(blocks) == 2
assert abs(blocks[0][3]["upload_packet_fraction"] - 0.5) < 1e-12
assert normalize_direction("outbound", "client") == "up"
assert normalize_direction("inbound", "server") == "up"
assert normalize_direction("outbound", "server") == "down"
print("packet fingerprinting synthetic test: PASS")
