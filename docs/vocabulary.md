# Personal vocabulary, aliases, and pronunciations

A local, user-owned store improves transcription, routing, and TTS pronunciation for
repository names, issue references, developer tools, acronyms, people, and recurring
speech-recognition mistakes. It is edited only through explicit
`voice-harness vocabulary` commands; the harness never silently learns a correction.

## Storage format and location

- A single JSON document (schema `version` 2) at
  `$XDG_CONFIG_HOME/voice-harness/vocabulary.json` (default
  `~/.config/voice-harness/vocabulary.json`). JSON keeps the harness free of runtime
  dependencies because it round-trips with only the standard library. The file is
  written privately (`0o600`) with sorted keys so backups and diffs are stable.
- Three entry kinds are stored. A **replacement** rewrites recognized text (`spoken`)
  to a corrected form (`written`). An **alias** maps a spoken `phrase` to a canonical
  entity `target`: an `owner/repo` repository (`kind` `repository`) or an
  `owner/repo#number` issue (`kind` `issue`). The kind is inferred from the target.
  A **pronunciation** maps a written project, organization, tool, or person name to a
  validated plain-text utterance used only in the spoken channel.

## Normalization

- Replacement sources and alias phrases are trimmed, have internal whitespace
  collapsed to single spaces, and are compared case-insensitively. Alias phrases are
  stored case-folded; written and target values keep their case.
- Matching against transcribed text is case-insensitive, whitespace-flexible, and only
  fires on whole phrases (never inside a longer word).
- Pronunciations are limited to 200 characters and plain words plus speech punctuation.
  Markup, shell syntax, and control characters are rejected.

## Precedence and conflict behavior

- STT corrections apply in order of user vocabulary first, then
  `DICTATION_REPLACEMENTS`, then built-in defaults. A user replacement overrides any
  static entry with the same spoken source.
- When resolving aliases, longer phrases match before shorter ones so the most
  specific alias wins.
- Each spoken source and each alias phrase maps to exactly one value. `add` rejects a
  conflicting key that already resolves to a different value unless `--force` is given,
  and a stored file containing the same key twice with different values is rejected as
  ambiguous on load.
- Pronunciations use the same conflict behavior and are applied longest-first. They
  never change displayed responses, stored job results, commands, or prompts.

Aliases are resolved as a deterministic pre-pass on the trusted utterance before the
intent router and repository/issue detection run, so `owner/repo` and
`owner/repo#number` references become available to existing routing. Vocabulary
content never leaves the machine; it only appears externally when a resolved alias or
corrected transcription becomes part of an agent prompt the user explicitly requested.

## Commands

```fish
voice-harness vocabulary list
voice-harness vocabulary list --kind alias
voice-harness vocabulary list --kind pronunciation
voice-harness vocabulary add replacement "herder" "herdr"
voice-harness vocabulary add alias "the harness repo" "joshua-mo-143/local-voice-harness"
voice-harness vocabulary add alias "harness bug" "joshua-mo-143/local-voice-harness#35"
voice-harness vocabulary add pronunciation "Herdr" "herder"
voice-harness vocabulary remove replacement "herder"
voice-harness vocabulary remove alias "the harness repo"
voice-harness vocabulary remove pronunciation "Herdr"
voice-harness pronounce "PR #128 changed src/http_client.py"
voice-harness vocabulary export --output vocabulary-backup.json
voice-harness vocabulary import vocabulary-backup.json          # merge
voice-harness vocabulary import vocabulary-backup.json --replace # overwrite
```

`export` without `--output` prints the JSON document to stdout for inspection or
piping to a backup. `import` merges by default (incoming entries win on conflict) or
replaces the whole store with `--replace`. Deleting an entry uses `remove`; deleting
the store entirely is a matter of removing the JSON file. The dictation service reads
the file on each transcription, and each foreground or wake process loads
pronunciations for its speech renderer. Restart a long-running wake process after
changing pronunciations. `voice-harness pronounce` previews the exact rendered text
without contacting the TTS service or playing audio.
