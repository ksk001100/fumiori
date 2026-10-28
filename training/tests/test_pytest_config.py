from pathlib import Path
import shutil
import subprocess
import sys


def test_tmp_path_works_without_build_directory(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for name in ('pytest.ini', 'conftest.py'):
        shutil.copyfile(root / name, tmp_path / name)
    (tmp_path / 'test_probe.py').write_text(
        'def test_tmp_path(tmp_path):\n'
        '    assert tmp_path.is_dir()\n',
        encoding='utf-8',
    )
    assert not (tmp_path / 'build').exists()

    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', 'test_probe.py'],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'build' / 'pytest-tmp').is_dir()
