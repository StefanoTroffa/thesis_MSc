import subprocess
import sys

def generate_requirements():
    python_version = sys.version.split(" ")[0]
    pip_freeze_output = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')
    
    with open('requirements.txt', 'w') as f:
        f.write(pip_freeze_output)

    with open('environment.yml', 'w') as f:
        f.write(f"""
name: myenv
channels:
  - defaults
dependencies:
  - python={python_version}
  - pip
""")
        for line in pip_freeze_output.splitlines():
            if '==' in line:
                package, version = line.split('==')
                f.write(f"  - {package}={version}\n")
            else:
                f.write(f"  - pip:\n")
                f.write(f"    - {line}\n")

if __name__ == "__main__":
    generate_requirements()
