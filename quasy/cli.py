import argparse
import json
from quasy.anonymizer import anonymize

def main():
    parser = argparse.ArgumentParser(description="QuASy: Anonymize queries")
    parser.add_argument("text", help="Input text to anonymize")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()
    safe, mapping = anonymize(args.text)

    if args.json:
        import json
        print(json.dumps({"safe": safe, "mapping": mapping}, indent=2))
    else: 
        print(f"Safe Text: {safe}")
        print(f"Mapping: {mapping}")

if __name__ == "__main__":
    main()