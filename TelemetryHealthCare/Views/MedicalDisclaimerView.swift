//
//  MedicalDisclaimerView.swift
//  TelemetryHealthCare
//
//  Created by Assistant on 2025-01-09.
//

import SwiftUI

struct MedicalDisclaimerView: View {
    @AppStorage("hasAcceptedDisclaimer") private var hasAcceptedDisclaimer = false
    @State private var showFullDisclaimer = false
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    VStack(spacing: 12) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 50))
                            .foregroundColor(.orange)
                            .accessibilityLabel("Warning icon")
                        
                        Text("Important Medical Disclaimer")
                            .font(.title)
                            .fontWeight(.bold)
                            .multilineTextAlignment(.center)
                            .accessibilityAddTraits(.isHeader)
                        
                        Text("Please read carefully before using Rhythm 360")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.top)
                    
                    // Main Disclaimer Content
                    VStack(alignment: .leading, spacing: 20) {
                        DisclaimerSection(
                            title: "Not a Medical Device",
                            icon: "stethoscope",
                            content: "Rhythm 360 is NOT a medical device and is NOT intended to diagnose, treat, cure, or prevent any disease or health condition. This app is for informational and wellness purposes only."
                        )
                        
                        DisclaimerSection(
                            title: "Not a Substitute for Medical Care",
                            icon: "cross.case.fill",
                            content: "The information provided by Rhythm 360 should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns."
                        )
                        
                        DisclaimerSection(
                            title: "Emergency Situations",
                            icon: "phone.fill",
                            content: "If you think you may have a medical emergency, call your doctor or emergency services (911 in the US) immediately. Do not rely on this app for emergency situations."
                        )
                        
                        DisclaimerSection(
                            title: "Accuracy Limitations",
                            icon: "exclamationmark.circle.fill",
                            content: "While our AI models have high accuracy in testing, they may produce incorrect results. Factors like device placement, movement, and individual physiology can affect readings. Never make medical decisions based solely on this app."
                        )
                        
                        DisclaimerSection(
                            title: "Consult Healthcare Providers",
                            icon: "person.2.fill",
                            content: "Always consult with your healthcare provider before making any changes to your medications, treatment plans, or lifestyle based on information from this app."
                        )
                        
                        DisclaimerSection(
                            title: "Data Privacy",
                            icon: "lock.fill",
                            content: "Your health data is processed locally on your device. We do not transmit health data to external servers. However, you are responsible for keeping your device secure."
                        )
                        
                        DisclaimerSection(
                            title: "Age Restrictions",
                            icon: "person.crop.circle.badge.exclamationmark",
                            content: "This app is intended for use by adults 18 years and older. It has not been evaluated for use in children or adolescents."
                        )
                        
                        DisclaimerSection(
                            title: "No Warranty",
                            icon: "xmark.shield.fill",
                            content: "This app is provided 'as is' without warranty of any kind. We do not guarantee uninterrupted or error-free operation."
                        )
                    }
                    .padding(.horizontal)
                    
                    // Legal Text
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Legal Notice")
                            .font(.headline)
                            .accessibilityAddTraits(.isHeader)
                        
                        Text("By using Rhythm 360, you acknowledge that you have read, understood, and agree to this disclaimer. The developers, distributors, and affiliates of Rhythm 360 shall not be liable for any direct, indirect, incidental, consequential, or punitive damages arising from your use of this app.")
                            .font(.caption)
                            .foregroundColor(.gray)
                        
                        Text("This app has not been evaluated by the Food and Drug Administration (FDA) or any other regulatory body. It is not CE marked for use in the European Union as a medical device.")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    .padding(.horizontal)
                    
                    // Accept Button
                    if !hasAcceptedDisclaimer {
                        Button(action: {
                            hasAcceptedDisclaimer = true
                            dismiss()
                        }) {
                            Text("I Understand and Accept")
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(12)
                        }
                        .padding(.horizontal)
                        .accessibilityLabel("Accept medical disclaimer")
                        .accessibilityHint("Tap to acknowledge and accept the medical disclaimer")
                    }
                    
                    // Version and Update Info
                    Text("Last Updated: January 2025 | Version 1.0")
                        .font(.caption2)
                        .foregroundColor(.gray)
                        .padding(.bottom)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if hasAcceptedDisclaimer {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button("Done") {
                            dismiss()
                        }
                    }
                }
            }
        }
    }
}

struct DisclaimerSection: View {
    let title: String
    let icon: String
    let content: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 30)
                .accessibilityHidden(true)
            
            VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.headline)
                    .accessibilityAddTraits(.isHeader)
                
                Text(content)
                    .font(.body)
                    .foregroundColor(.primary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title). \(content)")
    }
}

// MARK: - Inline Disclaimer Component
struct InlineDisclaimerView: View {
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "info.circle.fill")
                .foregroundColor(.blue)
                .accessibilityHidden(true)
            
            Text("For informational purposes only. Not medical advice.")
                .font(.caption)
                .foregroundColor(.gray)
        }
        .padding(8)
        .background(Color(.systemGray6))
        .cornerRadius(8)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Disclaimer: For informational purposes only. Not medical advice.")
    }
}

// MARK: - Emergency Banner Component
struct EmergencyBannerView: View {
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "phone.circle.fill")
                .font(.title2)
                .foregroundColor(.white)
                .accessibilityHidden(true)
            
            VStack(alignment: .leading, spacing: 2) {
                Text("Medical Emergency?")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                
                Text("Call 911 immediately")
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.9))
            }
            
            Spacer()
        }
        .padding(12)
        .background(Color.red)
        .cornerRadius(12)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Medical Emergency? Call 911 immediately")
        .accessibilityAddTraits(.isButton)
    }
}

// MARK: - App Launch Disclaimer Check
struct DisclaimerCheckModifier: ViewModifier {
    @AppStorage("hasAcceptedDisclaimer") private var hasAcceptedDisclaimer = false
    @State private var showDisclaimer = false
    
    func body(content: Content) -> some View {
        content
            .onAppear {
                if !hasAcceptedDisclaimer {
                    showDisclaimer = true
                }
            }
            .fullScreenCover(isPresented: $showDisclaimer) {
                MedicalDisclaimerView()
            }
    }
}

extension View {
    func withDisclaimerCheck() -> some View {
        modifier(DisclaimerCheckModifier())
    }
}