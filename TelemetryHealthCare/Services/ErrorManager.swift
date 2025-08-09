//
//  ErrorManager.swift
//  TelemetryHealthCare
//
//  Created by Assistant on 2025-01-09.
//

import Foundation
import SwiftUI
import os.log

// MARK: - Error Types
enum AppError: LocalizedError {
    case healthKitUnavailable
    case healthKitPermissionDenied
    case noHealthData
    case dataProcessingFailed(String)
    case modelPredictionFailed(String)
    case networkError(String)
    case storageError(String)
    case exportFailed(String)
    case unknown(String)
    
    var errorDescription: String? {
        switch self {
        case .healthKitUnavailable:
            return "Health data is not available on this device. Please ensure you have an Apple Watch paired."
        case .healthKitPermissionDenied:
            return "Permission to access health data was denied. Please enable access in Settings > Privacy > Health > Rhythm 360."
        case .noHealthData:
            return "No recent health data found. Please ensure your Apple Watch is collecting data."
        case .dataProcessingFailed(let reason):
            return "Unable to process health data: \(reason)"
        case .modelPredictionFailed(let reason):
            return "Health analysis failed: \(reason)"
        case .networkError(let reason):
            return "Network error: \(reason)"
        case .storageError(let reason):
            return "Storage error: \(reason)"
        case .exportFailed(let reason):
            return "Export failed: \(reason)"
        case .unknown(let reason):
            return "An unexpected error occurred: \(reason)"
        }
    }
    
    var recoverySuggestion: String? {
        switch self {
        case .healthKitUnavailable:
            return "Pair an Apple Watch with your iPhone to use this feature."
        case .healthKitPermissionDenied:
            return "Grant permission in Settings to enable health monitoring."
        case .noHealthData:
            return "Wear your Apple Watch for a few minutes and try again."
        case .dataProcessingFailed:
            return "Try refreshing the data or restart the app."
        case .modelPredictionFailed:
            return "Ensure you have sufficient recent health data and try again."
        case .networkError:
            return "Check your internet connection and try again."
        case .storageError:
            return "Free up storage space on your device."
        case .exportFailed:
            return "Check your permissions and try exporting again."
        case .unknown:
            return "Please restart the app. If the problem persists, contact support."
        }
    }
    
    var severity: ErrorSeverity {
        switch self {
        case .healthKitUnavailable, .healthKitPermissionDenied:
            return .critical
        case .noHealthData, .dataProcessingFailed, .modelPredictionFailed:
            return .warning
        case .networkError, .storageError, .exportFailed:
            return .error
        case .unknown:
            return .error
        }
    }
}

enum ErrorSeverity {
    case info
    case warning
    case error
    case critical
    
    var color: Color {
        switch self {
        case .info: return .blue
        case .warning: return .orange
        case .error: return .red
        case .critical: return .purple
        }
    }
    
    var icon: String {
        switch self {
        case .info: return "info.circle"
        case .warning: return "exclamationmark.triangle"
        case .error: return "exclamationmark.circle"
        case .critical: return "exclamationmark.octagon"
        }
    }
}

// MARK: - Error Manager
class ErrorManager: ObservableObject {
    static let shared = ErrorManager()
    
    @Published var currentError: AppError?
    @Published var showError = false
    @Published var errorHistory: [ErrorRecord] = []
    
    private let logger = Logger(subsystem: "com.rhythm360.app", category: "ErrorManager")
    private let maxErrorHistory = 100
    
    struct ErrorRecord: Identifiable {
        let id = UUID()
        let error: AppError
        let timestamp: Date
        let context: String?
        let stackTrace: String?
    }
    
    private init() {}
    
    // MARK: - Error Handling
    func handle(_ error: Error, context: String? = nil) {
        let appError: AppError
        
        if let err = error as? AppError {
            appError = err
        } else {
            appError = .unknown(error.localizedDescription)
        }
        
        logError(appError, context: context)
        recordError(appError, context: context)
        
        DispatchQueue.main.async {
            self.currentError = appError
            self.showError = true
        }
        
        // Send to crash reporting service
        CrashReporter.shared.recordError(error, context: context)
    }
    
    func handleSilent(_ error: Error, context: String? = nil) {
        let appError: AppError
        
        if let err = error as? AppError {
            appError = err
        } else {
            appError = .unknown(error.localizedDescription)
        }
        
        logError(appError, context: context)
        recordError(appError, context: context)
        
        // Send to crash reporting service without showing UI
        CrashReporter.shared.recordError(error, context: context)
    }
    
    // MARK: - Logging
    private func logError(_ error: AppError, context: String?) {
        let logMessage = """
        Error: \(error.localizedDescription ?? "Unknown error")
        Context: \(context ?? "None")
        Severity: \(error.severity)
        Recovery: \(error.recoverySuggestion ?? "None")
        """
        
        switch error.severity {
        case .info:
            logger.info("\(logMessage)")
        case .warning:
            logger.warning("\(logMessage)")
        case .error:
            logger.error("\(logMessage)")
        case .critical:
            logger.critical("\(logMessage)")
        }
    }
    
    // MARK: - Error History
    private func recordError(_ error: AppError, context: String?) {
        let record = ErrorRecord(
            error: error,
            timestamp: Date(),
            context: context,
            stackTrace: Thread.callStackSymbols.joined(separator: "\n")
        )
        
        DispatchQueue.main.async {
            self.errorHistory.insert(record, at: 0)
            if self.errorHistory.count > self.maxErrorHistory {
                self.errorHistory.removeLast()
            }
        }
    }
    
    func clearErrorHistory() {
        DispatchQueue.main.async {
            self.errorHistory.removeAll()
        }
    }
    
    func exportErrorLog() -> String {
        var log = "Rhythm 360 Error Log\n"
        log += "Generated: \(Date().formatted())\n\n"
        
        for record in errorHistory {
            log += "=====================================\n"
            log += "Time: \(record.timestamp.formatted())\n"
            log += "Error: \(record.error.localizedDescription ?? "Unknown")\n"
            log += "Severity: \(record.error.severity)\n"
            log += "Context: \(record.context ?? "None")\n"
            if let stack = record.stackTrace {
                log += "Stack Trace:\n\(stack)\n"
            }
            log += "\n"
        }
        
        return log
    }
}

// MARK: - Error View Component
struct ErrorAlertView: View {
    let error: AppError
    let onDismiss: () -> Void
    let onRetry: (() -> Void)?
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: error.severity.icon)
                .font(.system(size: 40))
                .foregroundColor(error.severity.color)
                .accessibilityLabel("Error icon")
            
            Text("Error")
                .font(.headline)
                .accessibilityAddTraits(.isHeader)
            
            Text(error.localizedDescription ?? "An error occurred")
                .font(.body)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            
            if let suggestion = error.recoverySuggestion {
                Text(suggestion)
                    .font(.caption)
                    .foregroundColor(.gray)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
            
            HStack(spacing: 16) {
                Button("Dismiss") {
                    onDismiss()
                }
                .buttonStyle(.bordered)
                
                if let retry = onRetry {
                    Button("Retry") {
                        retry()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 10)
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Error Handling View Modifier
struct ErrorHandling: ViewModifier {
    @ObservedObject var errorManager = ErrorManager.shared
    
    func body(content: Content) -> some View {
        content
            .alert("Error", isPresented: $errorManager.showError) {
                Button("OK") {
                    errorManager.showError = false
                }
            } message: {
                if let error = errorManager.currentError {
                    Text(error.localizedDescription ?? "An error occurred")
                    if let suggestion = error.recoverySuggestion {
                        Text(suggestion)
                    }
                }
            }
    }
}

extension View {
    func withErrorHandling() -> some View {
        modifier(ErrorHandling())
    }
}