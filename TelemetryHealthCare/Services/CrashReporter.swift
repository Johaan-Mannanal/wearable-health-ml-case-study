//
//  CrashReporter.swift
//  TelemetryHealthCare
//
//  Created by Assistant on 2025-01-09.
//

import Foundation
import UIKit
import os.log

// MARK: - Crash Reporter
class CrashReporter {
    static let shared = CrashReporter()
    
    private let logger = Logger(subsystem: "com.rhythm360.app", category: "CrashReporter")
    private let crashLogQueue = DispatchQueue(label: "com.rhythm360.crashreporter", qos: .background)
    
    private var sessionInfo: SessionInfo
    private var crashLogs: [CrashLog] = []
    private let maxCrashLogs = 50
    
    struct SessionInfo {
        let sessionId: UUID
        let appVersion: String
        let iosVersion: String
        let deviceModel: String
        let startTime: Date
        var crashCount: Int = 0
        var errorCount: Int = 0
    }
    
    struct CrashLog: Codable {
        let id: UUID
        let timestamp: Date
        let type: CrashType
        let message: String
        let context: String?
        let stackTrace: String?
        let sessionId: UUID
        let appVersion: String
        let iosVersion: String
        let deviceModel: String
        
        enum CrashType: String, Codable {
            case crash
            case error
            case warning
            case anr // Application Not Responding
        }
    }
    
    private init() {
        self.sessionInfo = SessionInfo(
            sessionId: UUID(),
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown",
            iosVersion: UIDevice.current.systemVersion,
            deviceModel: UIDevice.current.model,
            startTime: Date()
        )
        
        setupCrashHandlers()
        loadPreviousCrashLogs()
    }
    
    // MARK: - Crash Handling
    private func setupCrashHandlers() {
        // Set up exception handler
        NSSetUncaughtExceptionHandler { exception in
            CrashReporter.shared.handleCrash(exception: exception)
        }
        
        // Monitor app lifecycle for ANR detection
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(applicationDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(applicationWillTerminate),
            name: UIApplication.willTerminateNotification,
            object: nil
        )
    }
    
    private func handleCrash(exception: NSException) {
        let crashLog = CrashLog(
            id: UUID(),
            timestamp: Date(),
            type: .crash,
            message: exception.reason ?? "Unknown crash",
            context: exception.name.rawValue,
            stackTrace: exception.callStackSymbols.joined(separator: "\n"),
            sessionId: sessionInfo.sessionId,
            appVersion: sessionInfo.appVersion,
            iosVersion: sessionInfo.iosVersion,
            deviceModel: sessionInfo.deviceModel
        )
        
        saveCrashLog(crashLog)
        
        // In production, you would send this to a crash reporting service
        logger.critical("CRASH DETECTED: \(crashLog.message)")
    }
    
    // MARK: - Error Recording
    func recordError(_ error: Error, context: String? = nil) {
        crashLogQueue.async {
            self.sessionInfo.errorCount += 1
            
            let crashLog = CrashLog(
                id: UUID(),
                timestamp: Date(),
                type: .error,
                message: error.localizedDescription,
                context: context,
                stackTrace: Thread.callStackSymbols.joined(separator: "\n"),
                sessionId: self.sessionInfo.sessionId,
                appVersion: self.sessionInfo.appVersion,
                iosVersion: self.sessionInfo.iosVersion,
                deviceModel: self.sessionInfo.deviceModel
            )
            
            self.crashLogs.append(crashLog)
            if self.crashLogs.count > self.maxCrashLogs {
                self.crashLogs.removeFirst()
            }
            
            self.saveCrashLog(crashLog)
            
            // Log for debugging
            self.logger.error("Error recorded: \(crashLog.message), Context: \(context ?? "None")")
        }
    }
    
    func recordWarning(_ message: String, context: String? = nil) {
        crashLogQueue.async {
            let crashLog = CrashLog(
                id: UUID(),
                timestamp: Date(),
                type: .warning,
                message: message,
                context: context,
                stackTrace: nil,
                sessionId: self.sessionInfo.sessionId,
                appVersion: self.sessionInfo.appVersion,
                iosVersion: self.sessionInfo.iosVersion,
                deviceModel: self.sessionInfo.deviceModel
            )
            
            self.crashLogs.append(crashLog)
            if self.crashLogs.count > self.maxCrashLogs {
                self.crashLogs.removeFirst()
            }
            
            self.logger.warning("Warning recorded: \(message)")
        }
    }
    
    // MARK: - ANR Detection
    private var lastHeartbeat = Date()
    private var anrTimer: Timer?
    
    func startANRDetection() {
        anrTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
            let timeSinceLastHeartbeat = Date().timeIntervalSince(self.lastHeartbeat)
            if timeSinceLastHeartbeat > 10.0 {
                self.recordANR(duration: timeSinceLastHeartbeat)
            }
        }
    }
    
    func heartbeat() {
        lastHeartbeat = Date()
    }
    
    private func recordANR(duration: TimeInterval) {
        let crashLog = CrashLog(
            id: UUID(),
            timestamp: Date(),
            type: .anr,
            message: "Application not responding for \(Int(duration)) seconds",
            context: "Main thread blocked",
            stackTrace: Thread.callStackSymbols.joined(separator: "\n"),
            sessionId: sessionInfo.sessionId,
            appVersion: sessionInfo.appVersion,
            iosVersion: sessionInfo.iosVersion,
            deviceModel: sessionInfo.deviceModel
        )
        
        saveCrashLog(crashLog)
        logger.critical("ANR detected: \(crashLog.message)")
    }
    
    // MARK: - Persistence
    private func saveCrashLog(_ log: CrashLog) {
        crashLogQueue.async {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let crashLogsPath = documentsPath.appendingPathComponent("CrashLogs")
            
            // Create directory if needed
            try? FileManager.default.createDirectory(at: crashLogsPath, withIntermediateDirectories: true)
            
            let logPath = crashLogsPath.appendingPathComponent("\(log.id.uuidString).json")
            
            if let data = try? JSONEncoder().encode(log) {
                try? data.write(to: logPath)
            }
        }
    }
    
    private func loadPreviousCrashLogs() {
        crashLogQueue.async {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let crashLogsPath = documentsPath.appendingPathComponent("CrashLogs")
            
            guard let files = try? FileManager.default.contentsOfDirectory(at: crashLogsPath, includingPropertiesForKeys: nil) else {
                return
            }
            
            for file in files.prefix(self.maxCrashLogs) {
                if let data = try? Data(contentsOf: file),
                   let log = try? JSONDecoder().decode(CrashLog.self, from: data) {
                    self.crashLogs.append(log)
                }
            }
            
            self.logger.info("Loaded \(self.crashLogs.count) previous crash logs")
        }
    }
    
    // MARK: - Reporting
    func generateCrashReport() -> String {
        var report = "Rhythm 360 Crash Report\n"
        report += "========================\n\n"
        report += "Session Info:\n"
        report += "  Session ID: \(sessionInfo.sessionId)\n"
        report += "  App Version: \(sessionInfo.appVersion)\n"
        report += "  iOS Version: \(sessionInfo.iosVersion)\n"
        report += "  Device: \(sessionInfo.deviceModel)\n"
        report += "  Session Start: \(sessionInfo.startTime.formatted())\n"
        report += "  Error Count: \(sessionInfo.errorCount)\n"
        report += "  Crash Count: \(sessionInfo.crashCount)\n\n"
        
        report += "Recent Issues:\n"
        report += "==============\n"
        
        for log in crashLogs.suffix(20) {
            report += "\n[\(log.type.rawValue.uppercased())] \(log.timestamp.formatted())\n"
            report += "Message: \(log.message)\n"
            if let context = log.context {
                report += "Context: \(context)\n"
            }
            if let stack = log.stackTrace?.prefix(500) {
                report += "Stack: \(stack)...\n"
            }
            report += "---\n"
        }
        
        return report
    }
    
    func sendCrashReports() {
        // In production, this would send to a crash reporting service
        // For now, we'll just log that we would send them
        crashLogQueue.async {
            let unreportedLogs = self.crashLogs.filter { log in
                log.type == .crash || log.type == .anr
            }
            
            if !unreportedLogs.isEmpty {
                self.logger.info("Would send \(unreportedLogs.count) crash reports to server")
                // TODO: Implement actual crash reporting service integration
                // Options: Firebase Crashlytics, Sentry, Bugsnag, etc.
            }
        }
    }
    
    // MARK: - Lifecycle
    @objc private func applicationDidBecomeActive() {
        heartbeat()
        sendCrashReports()
    }
    
    @objc private func applicationWillTerminate() {
        logger.info("Application terminating normally")
    }
}