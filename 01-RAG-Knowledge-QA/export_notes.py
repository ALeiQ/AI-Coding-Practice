#!/usr/bin/env python3
"""Export Apple Notes to markdown files organized by folder."""

import os
import subprocess
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "备忘录")


def run_applescript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def get_folders() -> list[str]:
    script = """
    tell application "Notes"
        set folderNames to {}
        repeat with f in every folder
            set end of folderNames to name of f
        end repeat
        return folderNames
    end tell
    """
    raw = run_applescript(script)
    return [f.strip() for f in raw.split(", ") if f.strip()]


def get_notes_in_folder(folder_name: str) -> list[str]:
    script = f'''
    tell application "Notes"
        set noteNames to {{}}
        repeat with n in every note of folder "{folder_name}"
            set end of noteNames to name of n
        end repeat
        return noteNames
    end tell
    '''
    raw = run_applescript(script)
    return [n.strip() for n in raw.split(", ") if n.strip()]


def get_note_body(folder_name: str, note_name: str) -> str:
    escaped_name = note_name.replace('"', '\\"')
    escaped_folder = folder_name.replace('"', '\\"')
    script = f'''
    tell application "Notes"
        set noteBody to body of note "{escaped_name}" of folder "{escaped_folder}"
        return noteBody
    end tell
    '''
    return run_applescript(script)


def html_to_markdown(html: str) -> str:
    """Simple HTML to markdown conversion."""
    text = html
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"</p>", "", text)
    text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"# \1", text, flags=re.DOTALL)
    text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"## \1", text, flags=re.DOTALL)
    text = re.sub(r"<h3[^>]*>(.*?)</h3>", r"### \1", text, flags=re.DOTALL)
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(". ")
    return name[:200] if name else "untitled"


def main():
    folders = get_folders()
    total = 0

    for folder in folders:
        notes = get_notes_in_folder(folder)
        if not notes:
            continue

        folder_dir = os.path.join(DATA_DIR, sanitize_filename(folder))
        os.makedirs(folder_dir, exist_ok=True)

        for note_name in notes:
            try:
                body = get_note_body(folder, note_name)
                if not body.strip():
                    continue

                md_content = html_to_markdown(body)
                if not md_content.strip():
                    continue

                filename = sanitize_filename(note_name) + ".md"
                filepath = os.path.join(folder_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {note_name}\n\n")
                    f.write(md_content)

                total += 1
                print(f"  [{folder}] {note_name}")
            except Exception as e:
                print(f"  ERROR: {folder}/{note_name}: {e}")

    print(f"\nDone! Exported {total} notes to {DATA_DIR}")


if __name__ == "__main__":
    main()
