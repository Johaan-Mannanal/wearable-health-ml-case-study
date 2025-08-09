//
//  TelemetryHealthCareApp.swift
//  TelemetryHealthCare
//
//  Created by Yashwanth on 6/18/25.
//

import SwiftUI

@main
struct TelemetryHealthCareApp: App {
    init() {
        // Initialize crash reporting
        CrashReporter.shared.startANRDetection()
        
        // Initialize offline manager
        _ = OfflineManager.shared
        
        // Initialize error manager
        _ = ErrorManager.shared
    }
    
    var body: some Scene {
        WindowGroup {
            MainTabView()
                .withDisclaimerCheck()
                .withErrorHandling()
                .onAppear {
                    // Send crash reports if any from previous session
                    CrashReporter.shared.sendCrashReports()
                }
        }
    }
}
