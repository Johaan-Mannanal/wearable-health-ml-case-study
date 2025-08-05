import SwiftUI

struct ContentView: View {
    @State private var heartRates: [(Double, Date)] = []
    @State private var bloodPressures: [(systolic: Double, diastolic: Double, date: Date)] = []
    @State private var isLoading: Bool = false
    @State private var message: String = ""

    var body: some View {
        VStack(spacing: 20) {
            // Title
            Text("Penn State Telemetry")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.blue)
                .padding(.top)

            // Status Message
            if !message.isEmpty {
                Text(message)
                    .foregroundColor(.red)
                    .padding()
            } else if heartRates.isEmpty && bloodPressures.isEmpty {
                Text("No data yet. Tap buttons to fetch.")
                    .foregroundColor(.gray)
                    .italic()
            }

            // Scrollable Content
            ScrollView {
                // Heart Rate List
                if !heartRates.isEmpty {
                    Section(header: Text("Heart Rate").font(.headline)) {
                        ForEach(heartRates, id: \.1) { (rate, date) in
                            HStack {
                                Image(systemName: "heart.fill")
                                    .foregroundColor(.red)
                                Text("Heart Rate: \(String(format: "%.1f", rate)) bpm")
                                Spacer()
                                Text("\(date, style: .time)")
                                    .foregroundColor(.gray)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    .padding(.horizontal)
                }

                // Blood Pressure List
                if !bloodPressures.isEmpty {
                    Section(header: Text("Blood Pressure").font(.headline)) {
                        ForEach(bloodPressures, id: \.date) { bp in
                            HStack {
                                Image(systemName: "drop.fill")
                                    .foregroundColor(.purple)
                                Text("BP: \(String(format: "%.1f", bp.systolic))/\(String(format: "%.1f", bp.diastolic)) mmHg")
                                Spacer()
                                Text("\(bp.date, style: .time)")
                                    .foregroundColor(.gray)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    .padding(.horizontal)
                }
            }

            // Buttons
            HStack(spacing: 10) {
                Button(action: {
                    fetchHeartRate()
                }) {
                    Text(isLoading ? "Fetching..." : "Get Heart Rate")
                        .font(.headline)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(isLoading ? Color.gray : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .disabled(isLoading)

                Button(action: {
                    fetchBloodPressure()
                }) {
                    Text(isLoading ? "Fetching..." : "Get Blood Pressure")
                        .font(.headline)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(isLoading ? Color.gray : Color.purple)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .disabled(isLoading)
            }
            .padding(.horizontal)
        }
        .padding(.bottom)
        .onAppear {
            HealthKitManager.shared.askForPermission { success in
                DispatchQueue.main.async {
                    if !success {
                        message = "Permission denied. Check Settings > Health."
                    }
                }
            }
        }
    }

    func fetchHeartRate() {
        isLoading = true
        message = ""
        HealthKitManager.shared.getHeartRate { rates in
            DispatchQueue.main.async {
                isLoading = false
                if let rates = rates, !rates.isEmpty {
                    heartRates = rates
                } else {
                    message = "No heart rate data found."
                }
            }
        }
    }

    func fetchBloodPressure() {
        isLoading = true
        message = ""
        HealthKitManager.shared.getBloodPressure { bps in
            DispatchQueue.main.async {
                isLoading = false
                if let bps = bps, !bps.isEmpty {
                    bloodPressures = bps
                } else {
                    message = "No blood pressure data found."
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
