#!/usr/bin/env python3
"""Fail if tracked files, checkpoints, or Git history expose user identity."""

import argparse
import collections.abc
import os
import re
import subprocess
import sys

import torch
from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer


SENSITIVE_PATTERNS = {
  'absolute user-home path': re.compile(
    r'(?i)(?:^|[\s\'"])/(?:home|users)/[^/\s\'"]+'),
  'email address': re.compile(
    r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'),
}
TEXT_EXTENSIONS = {
  '.cfg', '.csv', '.json', '.md', '.py', '.sh', '.txt', '.yaml', '.yml',
}
ANONYMOUS_EMAIL = 'anonymous' + '@' + 'invalid'


def _git(root, *args):
  return subprocess.check_output(
    ['git', '-C', root, *args], text=True).strip()


def _find_sensitive(text, deny_tokens):
  text_for_patterns = text.replace(ANONYMOUS_EMAIL, '')
  findings = [
    label for label, pattern in SENSITIVE_PATTERNS.items()
    if pattern.search(text_for_patterns)
  ]
  lowered = text.lower()
  findings.extend(
    f'denied token {token!r}'
    for token in deny_tokens
    if token.lower() in lowered
  )
  return findings


def _walk(value, path='checkpoint', seen=None):
  if seen is None:
    seen = set()
  if isinstance(value, str):
    yield path, value
    return
  if value is None or isinstance(value, (bytes, int, float, bool)):
    return
  if torch.is_tensor(value):
    return
  object_id = id(value)
  if object_id in seen:
    return
  seen.add(object_id)
  if isinstance(value, BaseContainer):
    yield from _walk(
      OmegaConf.to_container(value, resolve=False), path, seen)
  elif isinstance(value, collections.abc.Mapping):
    for key, child in value.items():
      if isinstance(key, str):
        yield f'{path}.<key>', key
      yield from _walk(child, f'{path}[{key!r}]', seen)
  elif isinstance(value, (list, tuple, set)):
    for index, child in enumerate(value):
      yield from _walk(child, f'{path}[{index}]', seen)
  elif hasattr(value, '__dict__'):
    yield from _walk(vars(value), f'{path}.__dict__', seen)


def audit(root, deny_tokens):
  findings = []
  if root not in sys.path:
    sys.path.insert(0, root)
  tracked = _git(root, 'ls-files').splitlines()
  for relative_path in tracked:
    path = os.path.join(root, relative_path)
    extension = os.path.splitext(relative_path)[1].lower()
    if extension in TEXT_EXTENSIONS:
      with open(path, encoding='utf-8', errors='replace') as handle:
        for line_number, line in enumerate(handle, 1):
          for reason in _find_sensitive(line, deny_tokens):
            findings.append(
              f'{relative_path}:{line_number}: {reason}')
    elif extension == '.ckpt':
      checkpoint = torch.load(path, map_location='cpu')
      for value_path, text in _walk(checkpoint):
        for reason in _find_sensitive(text, deny_tokens):
          findings.append(
            f'{relative_path}:{value_path}: {reason}: {text!r}')

  history = _git(
    root, 'log', '--all',
    '--format=%H%x09%an%x09%ae%x09%cn%x09%ce').splitlines()
  for row in history:
    commit, author, author_email, committer, committer_email = row.split('\t')
    identity = f'{author} {author_email} {committer} {committer_email}'
    if (
        author != 'Anonymous'
        or committer != 'Anonymous'
        or author_email != ANONYMOUS_EMAIL
        or committer_email != ANONYMOUS_EMAIL):
      findings.append(f'{commit}: non-anonymous Git identity: {identity}')
    for reason in _find_sensitive(identity, deny_tokens):
      if reason.startswith('denied token'):
        findings.append(f'{commit}: {reason}')
  return findings


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', default='.')
  parser.add_argument(
    '--deny-token', action='append', default=[],
    help='Additional case-insensitive identity token to reject.')
  args = parser.parse_args()
  findings = audit(os.path.abspath(args.root), args.deny_token)
  if findings:
    print('\n'.join(findings), file=sys.stderr)
    raise SystemExit(1)
  print('Anonymity audit passed.')


if __name__ == '__main__':
  main()
