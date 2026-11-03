"""Versioned, hash-checked inference assets; no GT, network, or legacy experiment paths."""
import hashlib,json,os
from pathlib import Path


class Bundle:
    def __init__(self,folder=None):
        self.folder=Path(folder or os.environ.get('ANNOTATION_MODEL_DIR','/app/model_releases/auto_annotation/20261026.1')).resolve()
        self.manifest=json.loads((self.folder/'manifest.json').read_text())
        assert self.manifest['release']=='auto-annotation-20261026.1'
        self.settings=self.manifest['settings']
        for name,entry in self.manifest['assets'].items():
            path=self.path(name)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('ANNOTATION asset hash mismatch: '+name)
    def path(self,name):
        path=(self.folder/self.manifest['assets'][name]['file']).resolve()
        if not path.is_relative_to(self.folder):raise ValueError('Asset escapes model bundle')
        return path
    def document(self,name):return json.loads(self.path(name).read_text())
    def load(self,name):
        import torch
        return torch.jit.load(str(self.path(name)),map_location='cpu').eval()
