clc;
clear;
close all;

%% 1. Generate multiple synthetic calcium-like signals with labels (active/inactive)
% we have a total of fs*t samples
fs = 20;               % Sampling frequency
% its 20 Hz, which is commonly used in Calcium Imaging data because the changes are slow.
t = 0:1/fs:10;         % Time vector, 10 seconds
nTrials = 100;         % Number of trials (samples)

features = [];
labels = [];

for i = 1:nTrials
    % Behavior: 1 = active, 0 = inactive
    if rand > 0.5
        % Active behavior: more frequent calcium transients
        nSpikes = randi([20 40]);
        label = 1;
    else
        % Inactive behavior: fewer transients
        nSpikes = randi([5 15]);
        label = 0;
    end
    
    % Simulate calcium signal for this trial
    calcium_signal = zeros(size(t));
    spike_times = sort(rand(1, nSpikes) * max(t));
    decay_tau = 1; 
    
    for j = 1:length(spike_times)
        idx = find(t >= spike_times(j), 1);
        if ~isempty(idx)
            calcium_signal(idx:end) = calcium_signal(idx:end) + exp(-(t(1:end-idx+1) - t(1)) / decay_tau);
        end
    end
    
    drift = 0.1 * sin(2*pi*0.2*t);
    noise = 0.05 * randn(size(t));
    signal = calcium_signal + drift + noise;
    
    %% Bandpass filter        
    low_cutoff = 0.5;
    high_cutoff = 4;
    filter_order = 4;
    [b, a] = butter(filter_order, [low_cutoff, high_cutoff]/(fs/2), 'bandpass');
    filtered_signal = filtfilt(b, a, signal);
    
    % Feature 1: Bandpower between 0.5-4 Hz
    bp = bandpower(filtered_signal, fs, [0.5 4]);
    % (note for me):
    % The power of the frequency band 0.01 to 0.4 Hz is calculated, 
    % which is the range where slow brain activity is usually observed in Calcium Imaging data.    

    % Feature 2: Approximate Entropy
    m = 2;
    r = 0.2 * std(filtered_signal);
    apen = approximateEntropy(filtered_signal, m, r);    
    % This value indicates the level of disorder or complexity of the signal.
    % A value very close to zero means that the signal is very regular or predictable.
    % If this value were larger, it would indicate a signal with more complex and unpredictable changes.
    % Application:
    % In brain or physiological signals, approximate entropy helps us understand 
    % whether the brain or system in question is active in a normal state, or 
    % whether its activity has become simple and uniform due to disease or anesthesia.

    
    % Feature 1: captures the signal energy in the slow frequency range,
    % which relates to important brain activity and behavioral states.
    % Feature 2: Approximate Entropy measures the signal complexity or irregularity,
    % indicating how predictable or random the neural activity is.
    % Combining these features helps distinguish active vs. inactive states more effectively.

    features = [features; bp, apen];
    labels = [labels; label];
    
end

%% 2. Split data into training and testing
cv = cvpartition(length(labels), 'HoldOut', 0.3); % 30% is used for testing & 70% is used for training
XTrain = features(training(cv), :);
YTrain = labels(training(cv), :);
XTest = features(test(cv), :);
YTest = labels(test(cv), :);

%% 3. Train SVM classifier
SVMModel = fitcsvm(XTrain, YTrain);

%% 4. Test the model
YPred = predict(SVMModel, XTest);

%% 5. Classification accuracy
accuracy = sum(YPred == YTest) / length(YTest);
disp(['Classification Accuracy: ' num2str(accuracy*100) '%']);

%% Plot example signals of active and inactive (optional)
figure;
subplot(2,1,1);
plot(t, signal);
title('Example simulated calcium imaging signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
% This section plots the simulated calcium signal over time. 
% This graph shows what the raw signal looks like, including noise, slow oscillations, and calcium transients.

subplot(2,1,2);
gscatter(features(:,1), features(:,2), labels);
% gscatter(features(:,1), features(:,2), labels, ['b', 'r']); % blue for 0 (=Inactive), red for 1 (=Active)
xlabel('Bandpower (0.5-4 Hz)');
ylabel('Approximate Entropy');
title('Feature space (Bandpower vs Approximate Entropy)');
legend('Inactive','Active');

% This section shows the feature space used to train and test the classification model.
% Horizontal axis: Bandpower 0.5 to 4 Hz
% Vertical axis: Approximate Entropy
% Each point is a signal.
% The color of the points indicates the class label:
% One color for "Active"
% One color for "Inactive"
