# Campaign runbook

Run the rows of `schedule.csv` in order. For each run, copy its folder to every machine
(or share it), then start the server, then the proxy (which waits for the server's run
ID with --wait-for-run, so the same proxy config serves every run), then the clients.

```bash
python3 main.py run --config <run>/server.yaml          # jetson_agx_orin  (start first)
python3 main.py proxy --config <proxy config> --wait-for-run   # proxy host; picks up the run ID
python3 main.py run --config <run>/client_1.yaml        # jetson_orin_nano
python3 main.py run --config <run>/client_2.yaml        # dell_desktop
python3 main.py run --config <run>/client_3.yaml        # dell_laptop
```

Set `capture.extract_after: false` in the proxy config, so the proxy only captures and
finishes within minutes of the run ending. After the run is collected, copy the proxy's
pcap chunks to the DGX, extract them there in parallel, and check the run before
starting the next one:

```bash
rsync -av --progress <user>@<proxy-host>:<repo>/experiments/staging/<run_id>/proxy/pcap/ \
      /data/pcaps/<run_id>/
python3 extract_pcaps_parallel.py collected_experiments/<run_id> --pcap-dir /data/pcaps/<run_id>
python3 check_run.py collected_experiments/<run_id> --plan campaigns/custom_main/schedule.csv
```

Once the check passes, delete the run's pcap chunks on the proxy laptop to free its disk.

When the campaign is collected, build the dataset (X, Y, the single-client compute
features and the experiment ratios) in one step:

```bash
python3 prepare_fingerprinting_dataset.py
```

Re-run immediately if the check reports FAIL (give the repeat a new experiment number,
because the failed run's output folders already exist). Each role refuses to start on
the wrong kind of machine (Jetson / Linux / Windows), on a host other than
`campaign.expected_host` when set, or with an unsynchronized clock when
`campaign.require_clock_sync` is true; `AIFP_SKIP_DEVICE_CHECK=1` overrides the check.

| order | variant | dataset | rounds | experiment | config folder |
|---|---|---|---|---|---|
| 1 | bert_base | imdb | 50 | exp2 | `campaigns/custom_main/runs/001_bert_base_imdb_exp2` |
| 2 | distilbert_2layer | ag_news | 50 | exp1 | `campaigns/custom_main/runs/002_distilbert_2layer_ag_news_exp1` |
| 3 | bigru_2layer | uci_har | 50 | exp2 | `campaigns/custom_main/runs/003_bigru_2layer_uci_har_exp2` |
| 4 | resnet101 | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/004_resnet101_fashion_mnist_exp2` |
| 5 | bert_mini | ag_news | 50 | exp2 | `campaigns/custom_main/runs/005_bert_mini_ag_news_exp2` |
| 6 | convolutional_autoencoder_4layer | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/006_convolutional_autoencoder_4layer_fashion_mnist_exp2` |
| 7 | distilbert_base | ag_news | 50 | exp2 | `campaigns/custom_main/runs/007_distilbert_base_ag_news_exp2` |
| 8 | convolutional_autoencoder_2layer | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/008_convolutional_autoencoder_2layer_fashion_mnist_exp2` |
| 9 | efficientnet_b0 | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/009_efficientnet_b0_fashion_mnist_exp2` |
| 10 | mobilenet_v2 | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/010_mobilenet_v2_fashion_mnist_exp2` |
| 11 | tiny_transformer_2layer | ag_news | 50 | exp2 | `campaigns/custom_main/runs/011_tiny_transformer_2layer_ag_news_exp2` |
| 12 | tiny_transformer_6layer | ag_news | 50 | exp2 | `campaigns/custom_main/runs/012_tiny_transformer_6layer_ag_news_exp2` |
| 13 | gru_1layer | uci_har | 50 | exp2 | `campaigns/custom_main/runs/013_gru_1layer_uci_har_exp2` |
| 14 | bert_tiny | ag_news | 50 | exp2 | `campaigns/custom_main/runs/014_bert_tiny_ag_news_exp2` |
| 15 | efficientnet_b1 | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/015_efficientnet_b1_fashion_mnist_exp2` |
| 16 | distilbert_4layer | ag_news | 50 | exp1 | `campaigns/custom_main/runs/016_distilbert_4layer_ag_news_exp1` |
| 17 | efficientnet_b2 | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/017_efficientnet_b2_fashion_mnist_exp2` |
| 18 | resnet50 | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/018_resnet50_fashion_mnist_exp2` |
| 19 | mobilenet_v3_small | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/019_mobilenet_v3_small_fashion_mnist_exp2` |
| 20 | resnet34 | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/020_resnet34_fashion_mnist_exp2` |
| 21 | dense_autoencoder_3layer | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/021_dense_autoencoder_3layer_fashion_mnist_exp2` |
| 22 | vae_fc | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/022_vae_fc_fashion_mnist_exp2` |
| 23 | lstm_1layer | uci_har | 50 | exp2 | `campaigns/custom_main/runs/023_lstm_1layer_uci_har_exp2` |
| 24 | bert_small | ag_news | 50 | exp2 | `campaigns/custom_main/runs/024_bert_small_ag_news_exp2` |
| 25 | tiny_transformer_4layer | ag_news | 50 | exp2 | `campaigns/custom_main/runs/025_tiny_transformer_4layer_ag_news_exp2` |
| 26 | lstm_2layer | uci_har | 50 | exp2 | `campaigns/custom_main/runs/026_lstm_2layer_uci_har_exp2` |
| 27 | dense_autoencoder_2layer | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/027_dense_autoencoder_2layer_fashion_mnist_exp2` |
| 28 | convolutional_autoencoder_6layer | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/028_convolutional_autoencoder_6layer_fashion_mnist_exp2` |
| 29 | bilstm_2layer | uci_har | 50 | exp2 | `campaigns/custom_main/runs/029_bilstm_2layer_uci_har_exp2` |
| 30 | gru_2layer | uci_har | 50 | exp2 | `campaigns/custom_main/runs/030_gru_2layer_uci_har_exp2` |
| 31 | vae_conv | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/031_vae_conv_fashion_mnist_exp2` |
| 32 | dense_autoencoder_5layer | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/032_dense_autoencoder_5layer_fashion_mnist_exp2` |
| 33 | resnet18 | fashion_mnist | 50 | exp2 | `campaigns/custom_main/runs/033_resnet18_fashion_mnist_exp2` |
| 34 | resnet34 | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/034_resnet34_fashion_mnist_exp3` |
| 35 | bigru_2layer | uci_har | 50 | exp3 | `campaigns/custom_main/runs/035_bigru_2layer_uci_har_exp3` |
| 36 | dense_autoencoder_5layer | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/036_dense_autoencoder_5layer_fashion_mnist_exp3` |
| 37 | bert_base | imdb | 50 | exp3 | `campaigns/custom_main/runs/037_bert_base_imdb_exp3` |
| 38 | efficientnet_b2 | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/038_efficientnet_b2_fashion_mnist_exp3` |
| 39 | vae_fc | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/039_vae_fc_fashion_mnist_exp3` |
| 40 | mobilenet_v3_small | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/040_mobilenet_v3_small_fashion_mnist_exp3` |
| 41 | distilbert_base | ag_news | 50 | exp3 | `campaigns/custom_main/runs/041_distilbert_base_ag_news_exp3` |
| 42 | convolutional_autoencoder_2layer | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/042_convolutional_autoencoder_2layer_fashion_mnist_exp3` |
| 43 | lstm_2layer | uci_har | 50 | exp3 | `campaigns/custom_main/runs/043_lstm_2layer_uci_har_exp3` |
| 44 | tiny_transformer_4layer | ag_news | 50 | exp3 | `campaigns/custom_main/runs/044_tiny_transformer_4layer_ag_news_exp3` |
| 45 | dense_autoencoder_2layer | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/045_dense_autoencoder_2layer_fashion_mnist_exp3` |
| 46 | tiny_transformer_6layer | ag_news | 50 | exp3 | `campaigns/custom_main/runs/046_tiny_transformer_6layer_ag_news_exp3` |
| 47 | lstm_1layer | uci_har | 50 | exp3 | `campaigns/custom_main/runs/047_lstm_1layer_uci_har_exp3` |
| 48 | tiny_transformer_2layer | ag_news | 50 | exp3 | `campaigns/custom_main/runs/048_tiny_transformer_2layer_ag_news_exp3` |
| 49 | gru_2layer | uci_har | 50 | exp3 | `campaigns/custom_main/runs/049_gru_2layer_uci_har_exp3` |
| 50 | mobilenet_v2 | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/050_mobilenet_v2_fashion_mnist_exp3` |
| 51 | bilstm_2layer | uci_har | 50 | exp3 | `campaigns/custom_main/runs/051_bilstm_2layer_uci_har_exp3` |
| 52 | bert_small | ag_news | 50 | exp3 | `campaigns/custom_main/runs/052_bert_small_ag_news_exp3` |
| 53 | gru_1layer | uci_har | 50 | exp3 | `campaigns/custom_main/runs/053_gru_1layer_uci_har_exp3` |
| 54 | vae_conv | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/054_vae_conv_fashion_mnist_exp3` |
| 55 | distilbert_4layer | ag_news | 50 | exp2 | `campaigns/custom_main/runs/055_distilbert_4layer_ag_news_exp2` |
| 56 | resnet18 | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/056_resnet18_fashion_mnist_exp3` |
| 57 | bert_tiny | ag_news | 50 | exp3 | `campaigns/custom_main/runs/057_bert_tiny_ag_news_exp3` |
| 58 | efficientnet_b1 | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/058_efficientnet_b1_fashion_mnist_exp3` |
| 59 | convolutional_autoencoder_6layer | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/059_convolutional_autoencoder_6layer_fashion_mnist_exp3` |
| 60 | efficientnet_b0 | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/060_efficientnet_b0_fashion_mnist_exp3` |
| 61 | dense_autoencoder_3layer | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/061_dense_autoencoder_3layer_fashion_mnist_exp3` |
| 62 | bert_mini | ag_news | 50 | exp3 | `campaigns/custom_main/runs/062_bert_mini_ag_news_exp3` |
| 63 | resnet101 | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/063_resnet101_fashion_mnist_exp3` |
| 64 | resnet50 | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/064_resnet50_fashion_mnist_exp3` |
| 65 | distilbert_2layer | ag_news | 50 | exp2 | `campaigns/custom_main/runs/065_distilbert_2layer_ag_news_exp2` |
| 66 | convolutional_autoencoder_4layer | fashion_mnist | 50 | exp3 | `campaigns/custom_main/runs/066_convolutional_autoencoder_4layer_fashion_mnist_exp3` |
| 67 | efficientnet_b0 | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/067_efficientnet_b0_fashion_mnist_exp4` |
| 68 | vae_conv | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/068_vae_conv_fashion_mnist_exp4` |
| 69 | resnet50 | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/069_resnet50_fashion_mnist_exp4` |
| 70 | tiny_transformer_2layer | ag_news | 50 | exp4 | `campaigns/custom_main/runs/070_tiny_transformer_2layer_ag_news_exp4` |
| 71 | bert_mini | ag_news | 50 | exp4 | `campaigns/custom_main/runs/071_bert_mini_ag_news_exp4` |
| 72 | mobilenet_v2 | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/072_mobilenet_v2_fashion_mnist_exp4` |
| 73 | tiny_transformer_6layer | ag_news | 50 | exp4 | `campaigns/custom_main/runs/073_tiny_transformer_6layer_ag_news_exp4` |
| 74 | bert_small | ag_news | 50 | exp4 | `campaigns/custom_main/runs/074_bert_small_ag_news_exp4` |
| 75 | efficientnet_b1 | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/075_efficientnet_b1_fashion_mnist_exp4` |
| 76 | distilbert_4layer | ag_news | 50 | exp3 | `campaigns/custom_main/runs/076_distilbert_4layer_ag_news_exp3` |
| 77 | resnet34 | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/077_resnet34_fashion_mnist_exp4` |
| 78 | distilbert_2layer | ag_news | 50 | exp3 | `campaigns/custom_main/runs/078_distilbert_2layer_ag_news_exp3` |
| 79 | convolutional_autoencoder_4layer | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/079_convolutional_autoencoder_4layer_fashion_mnist_exp4` |
| 80 | convolutional_autoencoder_2layer | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/080_convolutional_autoencoder_2layer_fashion_mnist_exp4` |
| 81 | convolutional_autoencoder_6layer | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/081_convolutional_autoencoder_6layer_fashion_mnist_exp4` |
| 82 | dense_autoencoder_2layer | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/082_dense_autoencoder_2layer_fashion_mnist_exp4` |
| 83 | dense_autoencoder_5layer | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/083_dense_autoencoder_5layer_fashion_mnist_exp4` |
| 84 | mobilenet_v3_small | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/084_mobilenet_v3_small_fashion_mnist_exp4` |
| 85 | resnet18 | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/085_resnet18_fashion_mnist_exp4` |
| 86 | lstm_2layer | uci_har | 50 | exp4 | `campaigns/custom_main/runs/086_lstm_2layer_uci_har_exp4` |
| 87 | resnet101 | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/087_resnet101_fashion_mnist_exp4` |
| 88 | tiny_transformer_4layer | ag_news | 50 | exp4 | `campaigns/custom_main/runs/088_tiny_transformer_4layer_ag_news_exp4` |
| 89 | distilbert_base | ag_news | 50 | exp4 | `campaigns/custom_main/runs/089_distilbert_base_ag_news_exp4` |
| 90 | dense_autoencoder_3layer | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/090_dense_autoencoder_3layer_fashion_mnist_exp4` |
| 91 | bigru_2layer | uci_har | 50 | exp4 | `campaigns/custom_main/runs/091_bigru_2layer_uci_har_exp4` |
| 92 | gru_2layer | uci_har | 50 | exp4 | `campaigns/custom_main/runs/092_gru_2layer_uci_har_exp4` |
| 93 | bert_tiny | ag_news | 50 | exp4 | `campaigns/custom_main/runs/093_bert_tiny_ag_news_exp4` |
| 94 | lstm_1layer | uci_har | 50 | exp4 | `campaigns/custom_main/runs/094_lstm_1layer_uci_har_exp4` |
| 95 | gru_1layer | uci_har | 50 | exp4 | `campaigns/custom_main/runs/095_gru_1layer_uci_har_exp4` |
| 96 | bert_base | imdb | 50 | exp4 | `campaigns/custom_main/runs/096_bert_base_imdb_exp4` |
| 97 | efficientnet_b2 | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/097_efficientnet_b2_fashion_mnist_exp4` |
| 98 | bilstm_2layer | uci_har | 50 | exp4 | `campaigns/custom_main/runs/098_bilstm_2layer_uci_har_exp4` |
| 99 | vae_fc | fashion_mnist | 50 | exp4 | `campaigns/custom_main/runs/099_vae_fc_fashion_mnist_exp4` |
