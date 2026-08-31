from ai_fingerprint.result_collection import load_collector_settings, serve_collector_forever


if __name__ == "__main__":
    serve_collector_forever(load_collector_settings())
