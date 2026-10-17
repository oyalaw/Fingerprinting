# One-time validation capture (pilot only)

Purpose: show, once, that the proxy sees everything a client sends and receives and that its
timing is faithful. This is a check of the measurement method for the paper, not part of the
campaign. Do it in one pilot run, on one client, then never again.

## Which client

The Linux desktop (client_2, 10.42.0.47) is the best choice: capturing costs it little CPU, so
it does not change the compute gap being measured. Avoid the Jetson for this.

## Steps

1. Pick a medium model for the pilot run (e.g. MobileNetV2 or BERT-Tiny), so the capture is a
   few GB at most.

2. On the client, find the Wi-Fi interface (`ip -br addr`), then start a headers-only capture of
   its FL connection just before starting the client role:

   ```bash
   sudo dumpcap -i <wifi-interface> -s 256 -f "host 10.42.0.1 and tcp port 8080" -w ~/client2_check.pcapng
   ```

   `-s 256` keeps headers only (like the proxy), the filter keeps only the FL connection to the
   proxy. Leave it running for the whole run.

3. Run the pilot normally (proxy, server, clients). When the client role finishes, stop the
   capture with Ctrl+C. dumpcap prints "Packets received/dropped"; note the dropped count.

4. Collect the run as usual, extract it on the DGX (`extract_pcaps_parallel.py`) and run
   `check_run.py`. Copy the client capture to the DGX:

   ```bash
   scp ~/client2_check.pcapng <user>@10.42.0.195:~/lawrence/Fingerprinting/
   ```

5. Compare on the DGX:

   ```bash
   python3 compare_client_capture.py --client-pcap client2_check.pcapng --client-ip 10.42.0.47 --run collected_experiments/<run_id> --client-id client_2
   ```

## What a good result looks like

| Line in the output | Expected |
|---|---|
| rounds rebuilt | the same number of rounds from both captures (the planned rounds) |
| down_bytes / up_bytes median relative difference | well below 0.1 % |
| upload-start timing residual | IQR of a few ms (Wi-Fi jitter); no growing drift |
| compute gap, proxy − client | small and stable (a few ms): the proxy sees the gap the client actually had |
| payload coverage | ≈ 100 % |

Report the numbers in the paper as a validation of the observation point (one sentence and,
if there is room, a small table). If bytes or rounds differ, check the proxy's
`capture_statistics` (dropped packets) and the Wi-Fi log of that run before collecting more.
