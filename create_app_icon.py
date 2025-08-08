#!/usr/bin/env python3
"""
Create app icon for Rhythm 360
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_app_icon():
    # Create icon at 1024x1024 (required for App Store)
    size = 1024
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create gradient background
    for y in range(size):
        # Gradient from deep blue to purple
        r = int(30 + (138 - 30) * (y / size))  # 30 -> 138
        g = int(50 + (43 - 50) * (y / size))   # 50 -> 43
        b = int(180 + (226 - 180) * (y / size)) # 180 -> 226
        draw.rectangle([(0, y), (size, y+1)], fill=(r, g, b, 255))
    
    # Draw heart shape with ECG wave
    center_x = size // 2
    center_y = size // 2 - 50
    
    # Draw stylized heart
    heart_size = 300
    
    # Heart shape using bezier curves (simplified)
    heart_points = []
    for t in np.linspace(0, 2*np.pi, 100):
        x = 16 * np.sin(t)**3
        y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        heart_points.append((
            center_x + x * heart_size / 32,
            center_y + y * heart_size / 32
        ))
    
    # Fill heart with gradient effect
    for i in range(10):
        offset = i * 2
        color_intensity = 255 - i * 15
        scaled_points = [
            (center_x + (x - center_x) * (1 - offset/100), 
             center_y + (y - center_y) * (1 - offset/100))
            for x, y in heart_points
        ]
        draw.polygon(scaled_points, fill=(255, color_intensity, color_intensity, 255))
    
    # Draw ECG wave across the heart
    wave_points = []
    wave_y = center_y
    amplitude = 60
    
    # Create ECG pattern
    ecg_pattern = [0, 0, 5, -5, 0, 0, 0, -10, 40, -60, 20, 0, 0, 0, 5, 0, 0]
    pattern_width = len(ecg_pattern)
    
    for x in range(int(center_x - 250), int(center_x + 250), 10):
        idx = ((x - (center_x - 250)) // 10) % pattern_width
        y_offset = ecg_pattern[idx] * 2
        wave_points.append((x, wave_y + y_offset))
    
    # Draw ECG line with glow effect
    for offset in range(5, 0, -1):
        width = offset * 2
        alpha = 50 + (5 - offset) * 40
        color = (255, 255, 255, alpha)
        draw.line(wave_points, fill=color, width=width)
    
    # Draw main ECG line
    draw.line(wave_points, fill=(255, 255, 255, 255), width=4)
    
    # Add "360°" text at the bottom
    text = "360°"
    font_size = 120
    
    # Try different font paths
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        None  # Will skip if none work
    ]
    
    font = None
    for font_path in font_paths:
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
    
    # Calculate text position (approximate without font)
    text_x = center_x - 100  # Approximate width
    text_y = size - 200
    
    # Draw text with glow
    if font:
        for offset in range(3):
            glow_alpha = 100 - offset * 30
            draw.text((text_x - offset, text_y - offset), text, 
                     fill=(255, 255, 255, glow_alpha), font=font)
            draw.text((text_x + offset, text_y + offset), text, 
                     fill=(255, 255, 255, glow_alpha), font=font)
        
        draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    else:
        # Fallback: draw "360" without font
        draw.text((text_x, text_y), text, fill=(255, 255, 255, 255))
    
    # Add subtle circular border
    border_width = 20
    draw.ellipse([(border_width, border_width), 
                  (size - border_width, size - border_width)], 
                 outline=(255, 255, 255, 30), width=border_width)
    
    return img

def save_all_sizes(base_image):
    """Save icon in all required sizes for iOS"""
    
    sizes = [
        (20, 2, "icon-20@2x.png"),   # 40x40
        (20, 3, "icon-20@3x.png"),   # 60x60
        (29, 2, "icon-29@2x.png"),   # 58x58
        (29, 3, "icon-29@3x.png"),   # 87x87
        (40, 2, "icon-40@2x.png"),   # 80x80
        (40, 3, "icon-40@3x.png"),   # 120x120
        (60, 2, "icon-60@2x.png"),   # 120x120
        (60, 3, "icon-60@3x.png"),   # 180x180
        (1024, 1, "icon-1024.png"),  # 1024x1024
    ]
    
    output_dir = "/home/johaan/Documents/GitHub/TelemetryHealthCare/TelemetryHealthCare/Assets.xcassets/AppIcon.appiconset"
    
    for base_size, scale, filename in sizes:
        size = base_size * scale
        resized = base_image.resize((size, size), Image.LANCZOS)
        
        # Convert to RGB (remove alpha) for App Store icon
        if size == 1024:
            rgb_image = Image.new('RGB', (size, size), (255, 255, 255))
            rgb_image.paste(resized, (0, 0), resized)
            rgb_image.save(os.path.join(output_dir, filename), "PNG")
        else:
            resized.save(os.path.join(output_dir, filename), "PNG")
        
        print(f"Created {filename} ({size}x{size})")

if __name__ == "__main__":
    print("Creating Rhythm 360 app icon...")
    icon = create_app_icon()
    
    # Save the full resolution version
    icon.save("/home/johaan/Documents/GitHub/TelemetryHealthCare/TelemetryHealthCare/Assets.xcassets/AppIcon.appiconset/icon-1024.png", "PNG")
    
    # Save all required sizes
    save_all_sizes(icon)
    
    print("✅ App icon created successfully!")