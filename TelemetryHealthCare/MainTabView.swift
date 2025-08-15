//
//  MainTabView.swift
//  TelemetryHealthCare
//
//  Main navigation interface providing tab-based access to core app features.
//
//  This view serves as the primary navigation hub for the application,
//  organizing the three main functional areas:
//  1. AI Analysis - Real-time health monitoring and ML assessments
//  2. Trends - Historical data analysis and visualization
//  3. Settings - App configuration and data management
//
//  The tab bar design follows iOS Human Interface Guidelines for
//  intuitive navigation and consistent user experience.
//
//  Created by TelemetryHealthCare Team on 2024.
//

import SwiftUI

/// Main tab bar navigation controller for the TelemetryHealthCare app
///
/// Provides organized access to the three primary app sections through
/// a standard iOS tab bar interface. Maintains navigation state and
/// ensures consistent user experience across all sections.
///
/// - Note: Uses standard iOS tab bar patterns for familiar navigation
struct MainTabView: View {
    /// Currently selected tab index (0: Analysis, 1: Trends, 2: Settings)
    @State private var selectedTab = 0
    
    // MARK: - Tab Bar Interface
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Tab 1: AI Health Analysis
            AIAnalysisView()
                .tabItem {
                    Label("Analysis", systemImage: "waveform.path.ecg")
                }
                .tag(0)
            
            // Tab 2: Historical Trends
            TrendsView()
                .tabItem {
                    Label("Trends", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(1)
            
            // Tab 3: App Settings
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }
                .tag(2)
        }
        .accentColor(.blue)  // Consistent blue theme throughout app
    }
}

struct MainTabView_Previews: PreviewProvider {
    static var previews: some View {
        MainTabView()
    }
}