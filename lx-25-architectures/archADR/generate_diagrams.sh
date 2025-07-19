#!/bin/bash

# Generate PNG diagrams from PlantUML files
# This script processes all .puml files in the archADR directory

echo "🎨 Generating PNG diagrams from PlantUML files..."
echo "=============================================="

# Change to the archADR directory
cd "$(dirname "$0")"

# Create output directory for PNG files
mkdir -p png_output

# Counter for processed files
processed=0
failed=0

# Process each .puml file
for puml_file in *.puml; do
    # Skip if no .puml files found
    [ -e "$puml_file" ] || continue
    
    # Get filename without extension
    base_name="${puml_file%.puml}"
    
    echo -n "Processing $puml_file... "
    
    # Generate PNG using plantuml
    if /opt/homebrew/bin/plantuml -tpng -o png_output "$puml_file" 2>/dev/null; then
        echo "✅ Success"
        ((processed++))
    else
        echo "❌ Failed"
        ((failed++))
    fi
done

echo "=============================================="
echo "📊 Summary:"
echo "   Processed: $processed files"
echo "   Failed: $failed files"
echo "   Output directory: $(pwd)/png_output"
echo ""
echo "✨ Done! PNG files are in the png_output directory."