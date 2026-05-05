#!/usr/bin/env python3
"""
MedAI v1.1 — System Validation & Auto-Repair Script
Per Automation Operating Doctrine:
- Validates all prerequisites
- Auto-installs missing dependencies where possible
- Produces ONE executable remediation command for manual items
- No iterative debugging
"""

import sys
import os
import subprocess
import shutil
import platform
import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

class Validator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.base_dir = Path(__file__).parent
        self.platform = platform.system()
        self.install_commands = []
        
    def check(self, name: str, passed: bool, error_msg: str = "", warning: bool = False):
        """Record a validation check result"""
        if passed:
            print(f"{GREEN}✓{RESET} {name}")
            self.success_count += 1
        else:
            if warning:
                print(f"{YELLOW}⚠{RESET} {name}: {error_msg}")
                self.warnings.append(f"{name}: {error_msg}")
            else:
                print(f"{RED}✗{RESET} {name}: {error_msg}")
                self.errors.append(f"{name}: {error_msg}")
    
    def run_command(self, cmd: List[str], timeout: int = 10) -> Tuple[bool, str]:
        """Run a command and return success status and output"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=(self.platform == "Windows")
            )
            return result.returncode == 0, result.stdout + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return False, str(e)
    
    def auto_install_tesseract_windows(self) -> bool:
        """Attempt to download and provide installer for Tesseract on Windows"""
        installer_url = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
        installer_path = self.base_dir / "tesseract_installer.exe"
        
        try:
            print(f"{CYAN}→{RESET} Downloading Tesseract installer...")
            urllib.request.urlretrieve(installer_url, installer_path)
            print(f"{GREEN}✓{RESET} Downloaded to: {installer_path}")
            
            # Add to install commands
            self.install_commands.append({
                "tool": "Tesseract OCR",
                "command": f'"{installer_path}"',
                "note": "Run installer, use default settings, add to PATH when prompted"
            })
            return True
        except Exception as e:
            print(f"{RED}✗{RESET} Failed to download Tesseract: {e}")
            self.install_commands.append({
                "tool": "Tesseract OCR",
                "command": f"Download from: {installer_url}",
                "note": "Install manually and add to system PATH"
            })
            return False
    
    def check_python_version(self):
        """Validate Python version >= 3.11"""
        version = sys.version_info
        name = f"Python version (>= 3.11, found {version.major}.{version.minor})"
        passed = version >= (3, 11)
        
        if not passed:
            self.install_commands.append({
                "tool": "Python 3.11+",
                "command": "Download from: https://www.python.org/downloads/",
                "note": "Install Python 3.11 or higher, ensure 'Add to PATH' is checked"
            })
        
        self.check(name, passed, "Python 3.11+ required")
    
    def check_system_dependencies(self):
        """Check for tesseract and sqlcipher - auto-download if missing"""
        # Tesseract
        success, output = self.run_command(["tesseract", "--version"])
        
        if not success:
            print(f"{YELLOW}⚠{RESET} Tesseract OCR not found")
            
            if self.platform == "Windows":
                # Auto-download installer
                self.auto_install_tesseract_windows()
            elif self.platform == "Darwin":  # macOS
                self.install_commands.append({
                    "tool": "Tesseract OCR",
                    "command": "brew install tesseract",
                    "note": "Requires Homebrew (https://brew.sh)"
                })
            else:  # Linux
                self.install_commands.append({
                    "tool": "Tesseract OCR",
                    "command": "sudo apt-get install -y tesseract-ocr",
                    "note": "For Debian/Ubuntu systems"
                })
            
            self.errors.append("Tesseract OCR: Not installed")
        else:
            self.check("Tesseract OCR installed", True)
        
        # SQLCipher (warning only - Python package can substitute)
        success, _ = self.run_command(["sqlcipher", "-version"] if self.platform != "Windows" else ["where", "sqlcipher"])
        
        if not success:
            self.check(
                "SQLCipher installed",
                False,
                "Optional - Python sqlcipher3 package will be used instead",
                warning=True
            )
        else:
            self.check("SQLCipher installed", True)
    
    def check_file_structure(self):
        """Validate directory structure"""
        required_dirs = [
            "app", "ingestion", "extraction", "mkb", "orchestrator",
            "decision", "enrichment", "external_apis", "specialties",
            "tests", "scripts", "data"
        ]
        
        for dir_name in required_dirs:
            path = self.base_dir / dir_name
            self.check(
                f"Directory: {dir_name}/",
                path.exists() and path.is_dir(),
                f"Missing directory: {dir_name}"
            )
    
    def check_required_files(self):
        """Validate critical files exist"""
        required_files = [
            "Makefile",
            "requirements.txt",
            ".env.example",
            "app/main.py",
            "app/config.py",
            "mkb/sqlite_store.py",
            "decision/decision_engine.py",
            "tests/golden/__init__.py"
        ]
        
        for file_path in required_files:
            path = self.base_dir / file_path
            self.check(
                f"File: {file_path}",
                path.exists() and path.is_file(),
                f"Missing file: {file_path}"
            )
    
    def check_env_configuration(self):
        """Check .env file status"""
        env_example = self.base_dir / ".env.example"
        env_file = self.base_dir / ".env"
        
        self.check(
            ".env.example exists",
            env_example.exists(),
            "Missing .env.example template"
        )
        
        if env_file.exists():
            print(f"{BLUE}ℹ{RESET} .env file found (will validate API key during install)")
        else:
            print(f"{YELLOW}ℹ{RESET} .env not found (will be created from .env.example)")
    
    def check_data_directories(self):
        """Ensure data directories exist or can be created"""
        data_dirs = ["data/pdfs", "data/chroma", "data/pending"]
        
        for dir_path in data_dirs:
            full_path = self.base_dir / dir_path
            if not full_path.exists():
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    self.check(f"Created: {dir_path}/", True)
                except Exception as e:
                    self.check(f"Create: {dir_path}/", False, str(e))
            else:
                self.check(f"Exists: {dir_path}/", True)
    
    def check_python_imports(self):
        """Test critical imports (non-destructive)"""
        test_imports = [
            ("pathlib", "Path"),
            ("json", None),
            ("sqlite3", None),
        ]
        
        for module, attr in test_imports:
            try:
                mod = __import__(module)
                if attr:
                    getattr(mod, attr)
                self.check(f"Import: {module}", True)
            except ImportError as e:
                self.check(f"Import: {module}", False, str(e))
    
    def run_all_checks(self):
        """Run all validation checks"""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}MedAI v1.1 — System Validation{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
        
        print(f"{BLUE}[1/8]{RESET} Checking Python version...")
        self.check_python_version()
        
        print(f"\n{BLUE}[2/8]{RESET} Checking system dependencies...")
        self.check_system_dependencies()
        
        print(f"\n{BLUE}[3/8]{RESET} Checking file structure...")
        self.check_file_structure()
        
        print(f"\n{BLUE}[4/8]{RESET} Checking required files...")
        self.check_required_files()
        
        print(f"\n{BLUE}[5/8]{RESET} Checking environment configuration...")
        self.check_env_configuration()
        
        print(f"\n{BLUE}[6/8]{RESET} Checking/creating data directories...")
        self.check_data_directories()
        
        print(f"\n{BLUE}[7/8]{RESET} Checking Python standard library...")
        self.check_python_imports()
        
        print(f"\n{BLUE}[8/8]{RESET} Summary")
        self.print_summary()
    
    def print_summary(self):
        """Print final validation summary with consolidated next steps"""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{GREEN}✓ {self.success_count} checks passed{RESET}")
        
        if self.warnings:
            print(f"{YELLOW}⚠ {len(self.warnings)} warnings:{RESET}")
            for w in self.warnings:
                print(f"  • {w}")
        
        if self.errors:
            print(f"\n{RED}✗ {len(self.errors)} errors found{RESET}")
            
            if self.install_commands:
                print(f"\n{CYAN}{'='*70}{RESET}")
                print(f"{CYAN}AUTOMATED REMEDIATION — Execute these steps:{RESET}")
                print(f"{CYAN}{'='*70}{RESET}\n")
                
                for idx, cmd_info in enumerate(self.install_commands, 1):
                    print(f"{YELLOW}[{idx}]{RESET} {cmd_info['tool']}")
                    print(f"    Command: {CYAN}{cmd_info['command']}{RESET}")
                    if cmd_info.get('note'):
                        print(f"    Note: {cmd_info['note']}")
                    print()
                
                print(f"{CYAN}{'='*70}{RESET}")
                print(f"{YELLOW}After installation, re-run:{RESET} {CYAN}python validate_system.py{RESET}")
                print(f"{CYAN}{'='*70}{RESET}\n")
            else:
                for e in self.errors:
                    print(f"  • {e}")
            
            sys.exit(1)
        else:
            print(f"\n{GREEN}{'='*70}{RESET}")
            print(f"{GREEN}✓ SYSTEM READY FOR INSTALLATION{RESET}")
            print(f"{GREEN}{'='*70}{RESET}")
            print(f"\n{CYAN}NEXT STEPS (Execute in order):{RESET}\n")
            
            if self.platform == "Windows":
                print(f"{YELLOW}[1]{RESET} Create configuration file:")
                print(f"    {CYAN}copy .env.example .env{RESET}\n")
            else:
                print(f"{YELLOW}[1]{RESET} Create configuration file:")
                print(f"    {CYAN}cp .env.example .env{RESET}\n")
            
            print(f"{YELLOW}[2]{RESET} Edit .env file:")
            print(f"    • Set ANTHROPIC_API_KEY (get from: https://console.anthropic.com)")
            print(f"    • Set DB_ENCRYPTION_KEY (any strong passphrase)\n")
            
            print(f"{YELLOW}[3]{RESET} Install dependencies (~10 minutes):")
            if self.platform == "Windows":
                print(f"    {CYAN}python -m pip install -r requirements.txt{RESET}")
                print(f"    {CYAN}python -m spacy download en_core_web_trf{RESET}")
                print(f"    {CYAN}python scripts\\init_db.py{RESET}")
                print(f"    {CYAN}python scripts\\init_chroma.py{RESET}\n")
            else:
                print(f"    {CYAN}make install{RESET}\n")
            
            print(f"{YELLOW}[4]{RESET} Run golden tests (should show 25/25):")
            if self.platform == "Windows":
                print(f"    {CYAN}pytest tests\\golden\\ -v{RESET}\n")
            else:
                print(f"    {CYAN}make test-golden{RESET}\n")
            
            print(f"{YELLOW}[5]{RESET} Launch application:")
            if self.platform == "Windows":
                print(f"    {CYAN}streamlit run app\\main.py{RESET}\n")
            else:
                print(f"    {CYAN}make run{RESET}\n")
            
            print(f"{GREEN}{'='*70}{RESET}\n")

if __name__ == "__main__":
    validator = Validator()
    validator.run_all_checks()
