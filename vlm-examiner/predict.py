"""
Standalone prediction script for local testing.
Requires:  python predict.py <image_path>
           python predict.py <image_path> --config config.yaml --mock
"""
import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run MCQ solver on a single image")
    parser.add_argument("image_path", help="Path to the PNG image")
    parser.add_argument("--config",   default="config.yaml", help="Config YAML path")
    parser.add_argument("--mock",     action="store_true",
                        help="Mock the VLM (test fallback logic without GPU)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)

    if args.mock:
        # Patch VLM to return low-confidence directly (tests fallback chain)
        import unittest.mock as mock
        from src.vlm.inference import VLMResult
        mock_result = VLMResult(answer=None, confidence=0.1, raw_text="[MOCKED]")
        with mock.patch("src.vlm.inference.solve_direct", return_value=mock_result), \
             mock.patch("src.vlm.inference.solve_with_cot", return_value=mock_result), \
             mock.patch("src.vlm.inference.extract_text_and_math",
                        return_value={"question": "test", "options": {"1":"a","2":"b","3":"c","4":"d"},
                                      "has_math": False, "question_type": "conceptual"}):
            from src.pipeline import solve
            answer = solve(args.image_path, config_path=args.config)
    else:
        from src.pipeline import initialize_models, solve
        print("Initialising models (may take 2-5 minutes for 72B)...")
        try:
            initialize_models(args.config)
        except Exception as e:
            print(f"Init failed: {e}")
            print("Tip: run 'python scripts/download_hf_models.py' first.")
            sys.exit(1)
        answer = solve(args.image_path, config_path=args.config)

    print("\n" + "=" * 40)
    print(f"  PREDICTED ANSWER: {answer}")
    print("=" * 40)
    if answer == "5":
        print("  (Skipped — model was uncertain)")


if __name__ == "__main__":
    main()
