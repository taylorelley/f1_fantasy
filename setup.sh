#!/bin/bash
# setup.sh - Setup script for F1 Fantasy Optimizer Web App

echo "Setting up F1 Fantasy Optimizer Web Application..."

# Create directory structure
echo "Creating directories..."
mkdir -p templates
mkdir -p uploads
mkdir -p results

# Create a .gitignore file
echo "Creating .gitignore..."
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Flask
instance/
.webassets-cache

# Data
uploads/
results/
*.csv
*.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore
EOF

# Create a .dockerignore file
echo "Creating .dockerignore..."
cat > .dockerignore << EOF
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.gitignore
.mypy_cache
.pytest_cache
.hypothesis
uploads/
results/
*.csv
README.md
setup.sh
EOF

# Check if f1_optimizer.py exists
if [ ! -f "f1_optimizer.py" ]; then
    echo ""
    echo "WARNING: f1_optimizer.py not found!"
    echo "Please create f1_optimizer.py with the optimization code from the CLI version."
    echo "Remove the main() function and if __name__ == '__main__' block."
fi

# Compile SCSS to CSS if sass is installed
if command -v sass >/dev/null 2>&1; then
    echo "Compiling SCSS..."
    sass static/scss/style.scss static/style.css
else
    echo "Sass compiler not found. Skipping CSS build."
fi

# Make the script executable
chmod +x setup.sh

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Ensure f1_optimizer.py contains the optimization code"
echo "2. Place your CSV data files in the project directory"
echo "3. Run: docker-compose up --build"
echo "4. Open http://localhost:5000 in your browser"
echo ""
echo "For development without Docker:"
echo "1. Create virtual environment: python -m venv venv"
echo "2. Activate it: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
echo "3. Install dependencies: pip install -r requirements.txt"
echo "4. Run: python app.py"
