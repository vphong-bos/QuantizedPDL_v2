import argparse
import os
import onnx

def parse_args():
    parser = argparse.ArgumentParser(description="Merge ONNX external data into single file")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .onnx model",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output merged .onnx model",
    )

    return parser.parse_args()


def check_external_data(onnx_path):
    model_dir = os.path.dirname(onnx_path)

    # Look for any *.onnx_data file in same folder
    data_files = [f for f in os.listdir(model_dir) if f.endswith(".onnx_data")]

    if len(data_files) == 0:
        print("[WARNING] No .onnx_data file found in the same directory.")
        print("          If model uses external data, this may fail.")

    elif len(data_files) == 1:
        print(f"[INFO] Found external data file: {data_files[0]}")

    else:
        print(f"[INFO] Multiple .onnx_data files found: {data_files}")

    return data_files


def merge_onnx(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")

    print(f"[INFO] Checking external data...")
    check_external_data(input_path)

    print(f"[INFO] Loading model: {input_path}")
    model = onnx.load(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("[INFO] Merging external data into single ONNX file...")
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=False,
    )

    print(f"[DONE] Saved merged model to: {output_path}")


def main():
    args = parse_args()

    merge_onnx(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()