#!/usr/bin/env python3
"""
Generate a simple, minimal app icon for Rhythm 360
"""

from PIL import Image, ImageDraw
import os
import json

def create_simple_icon(size):
    """Create a simple heart icon with gradient background"""
    # Create image with gradient background
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Simple gradient background - blue to lighter blue
    for y in range(size):
        # Gradient from top to bottom
        ratio = y / size
        r = int(0 + (30 * ratio))  # 0 to 30
        g = int(122 + (40 * ratio))  # 122 to 162  
        b = int(255 - (50 * ratio))  # 255 to 205
        draw.rectangle([(0, y), (size, y+1)], fill=(r, g, b))
    
    # Draw a simple white heart in center
    center_x = size // 2
    center_y = size // 2
    heart_size = size * 0.4  # Heart takes up 40% of icon
    
    # Simple heart shape using circles and triangle
    # Left circle
    left_center = (center_x - heart_size * 0.25, center_y - heart_size * 0.1)
    # Right circle  
    right_center = (center_x + heart_size * 0.25, center_y - heart_size * 0.1)
    
    # Draw two circles for top of heart
    radius = heart_size * 0.3
    
    # Create a white heart by drawing circles and a triangle
    # Left circle
    draw.ellipse([
        left_center[0] - radius,
        left_center[1] - radius,
        left_center[0] + radius,
        left_center[1] + radius
    ], fill='white')
    
    # Right circle
    draw.ellipse([
        right_center[0] - radius,
        right_center[1] - radius,
        right_center[0] + radius,
        right_center[1] + radius
    ], fill='white')
    
    # Triangle for bottom of heart
    triangle = [
        (center_x - heart_size * 0.5, center_y),
        (center_x + heart_size * 0.5, center_y),
        (center_x, center_y + heart_size * 0.6)
    ]
    draw.polygon(triangle, fill='white')
    
    return img

def generate_icons():
    """Generate all required icon sizes"""
    
    # Icon sizes required by iOS
    sizes = [
        (20, 2),   # 20pt iPhone Notification @2x
        (20, 3),   # 20pt iPhone Notification @3x
        (29, 2),   # 29pt iPhone Settings @2x
        (29, 3),   # 29pt iPhone Settings @3x
        (40, 2),   # 40pt iPhone Spotlight @2x
        (40, 3),   # 40pt iPhone Spotlight @3x
        (60, 2),   # 60pt iPhone App @2x
        (60, 3),   # 60pt iPhone App @3x
        (1024, 1), # App Store
    ]
    
    # Create icons directory
    output_dir = "AppIcon"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each icon
    for base_size, scale in sizes:
        actual_size = base_size * scale
        img = create_simple_icon(actual_size)
        
        if base_size == 1024:
            filename = f"Icon-{base_size}.png"
        else:
            filename = f"Icon-{base_size}@{scale}x.png"
        
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, "PNG")
        print(f"Generated {filename} ({actual_size}x{actual_size})")
    
    # Create Contents.json for Asset Catalog
    contents = {
        "images": [
            {
                "size": "20x20",
                "idiom": "iphone",
                "filename": "Icon-20@2x.png",
                "scale": "2x"
            },
            {
                "size": "20x20",
                "idiom": "iphone",
                "filename": "Icon-20@3x.png",
                "scale": "3x"
            },
            {
                "size": "29x29",
                "idiom": "iphone",
                "filename": "Icon-29@2x.png",
                "scale": "2x"
            },
            {
                "size": "29x29",
                "idiom": "iphone",
                "filename": "Icon-29@3x.png",
                "scale": "3x"
            },
            {
                "size": "40x40",
                "idiom": "iphone",
                "filename": "Icon-40@2x.png",
                "scale": "2x"
            },
            {
                "size": "40x40",
                "idiom": "iphone",
                "filename": "Icon-40@3x.png",
                "scale": "3x"
            },
            {
                "size": "60x60",
                "idiom": "iphone",
                "filename": "Icon-60@2x.png",
                "scale": "2x"
            },
            {
                "size": "60x60",
                "idiom": "iphone",
                "filename": "Icon-60@3x.png",
                "scale": "3x"
            },
            {
                "size": "1024x1024",
                "idiom": "ios-marketing",
                "filename": "Icon-1024.png",
                "scale": "1x"
            }
        ],
        "info": {
            "version": 1,
            "author": "xcode"
        }
    }
    
    # Save Contents.json
    contents_path = os.path.join(output_dir, "Contents.json")
    with open(contents_path, 'w') as f:
        json.dump(contents, f, indent=2)
    
    print(f"\nAll icons generated in '{output_dir}' directory")
    print("Copy the entire 'AppIcon' folder to:")
    print("TelemetryHealthCare/Assets.xcassets/AppIcon.appiconset/")

if __name__ == "__main__":
    generate_icons()
    print("\nSimple icon generation complete!")