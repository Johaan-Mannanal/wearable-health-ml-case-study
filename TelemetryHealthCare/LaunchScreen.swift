//
//  LaunchScreen.swift
//  Rhythm 360
//
//  Animated launch screen for professional app appearance
//

import SwiftUI

struct LaunchScreen: View {
    @State private var isAnimating = false
    @State private var showApp = false
    
    var body: some View {
        if showApp {
            MainTabView()
        } else {
            ZStack {
                // Gradient Background
                LinearGradient(
                    colors: [
                        Color(red: 0.0, green: 0.48, blue: 1.0),
                        Color(red: 0.0, green: 0.3, blue: 0.8)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                VStack(spacing: 30) {
                    // Heart Icon with Pulse Animation
                    ZStack {
                        Circle()
                            .fill(.white.opacity(0.2))
                            .frame(width: 150, height: 150)
                            .scaleEffect(isAnimating ? 1.3 : 1.0)
                            .opacity(isAnimating ? 0 : 0.3)
                            .animation(
                                .easeInOut(duration: 1.5)
                                .repeatForever(autoreverses: false),
                                value: isAnimating
                            )
                        
                        Circle()
                            .fill(.white.opacity(0.15))
                            .frame(width: 120, height: 120)
                            .scaleEffect(isAnimating ? 1.2 : 1.0)
                            .opacity(isAnimating ? 0 : 0.5)
                            .animation(
                                .easeInOut(duration: 1.5)
                                .repeatForever(autoreverses: false)
                                .delay(0.2),
                                value: isAnimating
                            )
                        
                        Image(systemName: "heart.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.white)
                            .scaleEffect(isAnimating ? 1.1 : 1.0)
                            .animation(
                                .easeInOut(duration: 0.8)
                                .repeatForever(autoreverses: true),
                                value: isAnimating
                            )
                    }
                    
                    VStack(spacing: 10) {
                        Text("Rhythm 360")
                            .font(.system(size: 42, weight: .bold, design: .rounded))
                            .foregroundColor(.white)
                        
                        Text("AI-Powered Health Monitoring")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundColor(.white.opacity(0.9))
                    }
                    
                    // Loading Indicator
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(1.2)
                        .padding(.top, 30)
                }
                .onAppear {
                    isAnimating = true
                    // Transition to main app after 2.5 seconds
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                        withAnimation(.easeInOut(duration: 0.5)) {
                            showApp = true
                        }
                    }
                }
            }
        }
    }
}

struct LaunchScreen_Previews: PreviewProvider {
    static var previews: some View {
        LaunchScreen()
    }
}