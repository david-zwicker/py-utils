#!/usr/bin/env python3
"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import argparse
import pathlib
import re


def get_undefined_refs(logfile):
    """collect undefined references

    Args:
        logfile: The path to the latex logfile

    Returns:
        A list of bibtex keys that were not found in latex
    """
    with pathlib.Path(logfile).open() as f:
        first_line = f.readline()
        if "tex" not in first_line.lower():
            raise RuntimeError(f"{logfile} does not seem to be a LaTeX log file")

        results = []
        for line in f:
            matches = re.search("Citation `(\S+)' on page (?:\d+) undefined", line)
            if matches:
                results.append(matches.group(1))

    return results


def extract_refs(bibtex_file, ref_keys, ignore_fields=None):
    """extract specific references from a bibtex file

    Args:
        bibtex_file: The path to the bibtex file
        ref_keys: A sequence of keys that should be extracted

    Returns:
        dict: A dictionary of all bibtex entries that have been found. Note that some
        keys might not be included if they are not present in the database
    """
    import pybtex.database

    bib_data = pybtex.database.parse_file(pathlib.Path(bibtex_file))

    found = {}
    for key, value in bib_data.entries.items():
        if key in set(ref_keys):
            if ignore_fields is not None:
                delete_fields = [
                    field_name
                    for field_name in value.fields.keys()
                    if any(re.match(pattern, field_name) for pattern in ignore_fields)
                ]
                for field in delete_fields:
                    del value.fields[field]
            found[key] = value

    return pybtex.database.BibliographyData(found)


def copy_bibtex_to_clipboard(refs):
    """copies references to the clipboard

    Args:
        refs (dict): A dictionary of bibtex references
    """
    import pyperclip

    pyperclip.copy(refs.to_string("bibtex"))
    print("Copied bibtex data to clipboard...")


def main():
    """read command line arguments and call the parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", help="the logfile of the latex run")
    parser.add_argument("--library", help="the bibfile where references are read from")
    parser.add_argument("--output", help="the output bibtex file")
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="remove extra information from records",
    )
    parser.add_argument(
        "--clipboard",
        action="store_true",
        help="copy bibtex data to the clipboard",
    )

    args = parser.parse_args()

    # obtain the undefined references
    undefined_refs = get_undefined_refs(args.logfile)

    if undefined_refs:
        print("Undefined references: " + ", ".join(sorted(undefined_refs)))

        if args.library:
            # search the entries in the library
            if args.minimal:
                ignore_fields = ["abstract", "bdsk-.*", "keywords", "local-url"]
            else:
                ignore_fields = None

            extracted_refs = extract_refs(
                args.library,
                ref_keys=undefined_refs,
                ignore_fields=ignore_fields,
            )

            if args.clipboard:
                copy_bibtex_to_clipboard(extracted_refs)
            if args.output:
                extracted_refs.to_file(args.output)

            dangling_refs = set(undefined_refs) - set(extracted_refs.entries.keys())
            if dangling_refs:
                print(
                    "References not in the library: " + ", ".join(sorted(dangling_refs))
                )

    else:
        print("All references were defined!")


if __name__ == "__main__":
    main()
