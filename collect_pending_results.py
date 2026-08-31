from ai_fingerprint.result_collection import resend_pending_results


if __name__ == "__main__":
    results = resend_pending_results("experiments")
    if not results:
        print("No pending completed results were found.")
    else:
        print("\nCollection retry summary:")
        for item in results:
            print(f"  {item['config']}: {item.get('status')}")
