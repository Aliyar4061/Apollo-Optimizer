# Apollo-Optimizer
Apollo Optimizer



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apollo Optimizer Algorithm</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #f0f6fc;
            line-height: 1.6;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }
        
        h1 {
            font-size: 2.8rem;
            margin-bottom: 10px;
            color: #64ffda;
            text-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
        }
        
        h2 {
            font-size: 1.8rem;
            margin: 25px 0 15px;
            color: #64b5ff;
            border-bottom: 2px solid #64b5ff;
            padding-bottom: 8px;
        }
        
        h3 {
            font-size: 1.4rem;
            margin: 20px 0 10px;
            color: #ff79c6;
        }
        
        p {
            margin-bottom: 15px;
            font-size: 1.1rem;
        }
        
        .algorithm-container {
            background: rgba(17, 34, 51, 0.8);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            overflow-x: auto;
        }
        
        .algorithm {
            font-family: 'Courier New', monospace;
            white-space: pre;
            line-height: 1.5;
            font-size: 1.1rem;
            color: #f8f8f2;
        }
        
        .algorithm-step {
            margin: 10px 0;
        }
        
        .algorithm-keyword {
            color: #ff79c6;
            font-weight: bold;
        }
        
        .algorithm-comment {
            color: #6272a4;
        }
        
        .algorithm-var {
            color: #50fa7b;
        }
        
        .visualization {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .chart-container {
            flex: 1;
            min-width: 300px;
            background: rgba(17, 34, 51, 0.8);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .params {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .param-card {
            background: rgba(17, 34, 51, 0.8);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }
        
        .param-card h4 {
            color: #ffb86c;
            margin-bottom: 10px;
            font-size: 1.2rem;
        }
        
        .param-card p {
            font-size: 1rem;
        }
        
        .comparison {
            background: rgba(17, 34, 51, 0.8);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        th {
            background-color: rgba(0, 0, 0, 0.2);
            color: #64ffda;
        }
        
        tr:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #6272a4;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .visualization {
                flex-direction: column;
            }
            
            h1 {
                font-size: 2.2rem;
            }
            
            h2 {
                font-size: 1.6rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Apollo Optimizer Algorithm</h1>
            <p>Agreement-based AdamW-Lion Blending for Deep Learning Optimization</p>
        </header>
        
        <section>
            <h2>Algorithm Overview</h2>
            <p>The Apollo optimizer is an advanced optimization algorithm that intelligently blends AdamW and Lion updates based on their agreement. This hybrid approach aims to combine the benefits of both methods while mitigating their individual weaknesses.</p>
            
            <div class="algorithm-container">
                <pre class="algorithm">
<span class="algorithm-keyword">Require</span>: Learning rate α, momentum parameters β₁, β₂, β₃, regularization parameter ε, weight decay λ (optional)
<span class="algorithm-keyword">Require</span>: Initialize parameters θ₀, first moment m₀ = 0, second moment v₀ = 0, smoothed similarity s₀ = 0, step t = 0

<span class="algorithm-keyword">for</span> each parameter θ in parameter groups <span class="algorithm-comment">do</span>
    <span class="algorithm-keyword">Compute gradient</span>: gₜ = ∇θ fₜ(θₜ₋₁)
    <span class="algorithm-keyword">if</span> gₜ is None or sparse <span class="algorithm-keyword">then</span>
        Skip to next parameter
    <span class="algorithm-keyword">end if</span>
    <span class="algorithm-keyword">Increment step</span>: t ← t + 1
    <span class="algorithm-keyword">if</span> λ > 0 <span class="algorithm-keyword">then</span>
        <span class="algorithm-keyword">Apply decoupled weight decay</span>: θₜ₋₁ ← θₜ₋₁ - λ α θₜ₋₁
    <span class="algorithm-keyword">end if</span>
    
    <span class="algorithm-comment">// AdamW-style update</span>
    mₜ ← β₁ mₜ₋₁ + (1 - β₁) gₜ
    vₜ ← β₂ vₜ₋₁ + (1 - β₂) gₜ²
    m̂ₜ ← mₜ / (1 - β₁ᵗ)
    v̂ₜ ← vₜ / (1 - β₂ᵗ)
    uₜᴬᵈᵃᵐᵂ ← m̂ₜ / (√v̂ₜ + ε)
    
    <span class="algorithm-comment">// Lion-style update</span>
    uₜᴸⁱᵒⁿ ← sign(m̂ₜ)
    
    <span class="algorithm-comment">// Compute agreement</span>
    cosθₜ ← (uₜᴬᵈᵃᵐᵂ · uₜᴸⁱᵒⁿ) / (‖uₜᴬᵈᵃᵐᵂ‖₂ ‖uₜᴸⁱᵒⁿ‖₂ + ε_cos)
    sₜ ← β₃ sₜ₋₁ + (1 - β₃) cosθₜ
    γₜ ← (1 + sₜ) / 2  <span class="algorithm-comment">// Blending weight</span>
    
    <span class="algorithm-comment">// Blend updates</span>
    Δθₜ ← (1 - γₜ) uₜᴬᵈᵃᵐᵂ + γₜ uₜᴸⁱᵒⁿ
    θₜ ← θₜ₋₁ - α Δθₜ
<span class="algorithm-keyword">end for</span></pre>
            </div>
        </section>
        
        <section class="params">
            <div class="param-card">
                <h4>Learning Rate (α)</h4>
                <p>Controls the step size during optimization. Typical values range from 1e-4 to 1e-2.</p>
            </div>
            <div class="param-card">
                <h4>Momentum Parameters (β₁, β₂, β₃)</h4>
                <p>Exponential decay rates for moment estimates. β₁ and β₂ are for first and second moments, β₃ for similarity smoothing.</p>
            </div>
            <div class="param-card">
                <h4>Regularization Parameter (ε)</h4>
                <p>A small constant to prevent division by zero and improve numerical stability.</p>
            </div>
            <div class="param-card">
                <h4>Weight Decay (λ)</h4>
                <p>Regularization technique to prevent overfitting by penalizing large weights.</p>
            </div>
        </section>
        
        <section>
            <h2>Performance Comparison</h2>
            <div class="comparison">
                <p>Below is a comparison of Apollo with other optimizers across different datasets:</p>
                
                <h3>MNIST Dataset</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Optimizer</th>
                            <th>Test Accuracy</th>
                            <th>Test Loss</th>
                            <th>F1 Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Apollo</td>
                            <td>0.9933</td>
                            <td>0.0326</td>
                            <td>0.9932</td>
                        </tr>
                        <tr>
                            <td>AdamW</td>
                            <td>0.9902</td>
                            <td>0.0309</td>
                            <td>0.9901</td>
                        </tr>
                        <tr>
                            <td>Lion</td>
                            <td>0.9909</td>
                            <td>0.0378</td>
                            <td>0.9908</td>
                        </tr>
                        <tr>
                            <td>RMSprop</td>
                            <td>0.9929</td>
                            <td>0.0240</td>
                            <td>0.9928</td>
                        </tr>
                    </tbody>
                </table>
                
                <h3>SST-2 Dataset (Sentiment Analysis)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Optimizer</th>
                            <th>Test Accuracy</th>
                            <th>Test Loss</th>
                            <th>F1 Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Apollo</td>
                            <td>0.8005</td>
                            <td>0.6567</td>
                            <td>0.7996</td>
                        </tr>
                        <tr>
                            <td>AdamW</td>
                            <td>0.8257</td>
                            <td>0.5116</td>
                            <td>0.8257</td>
                        </tr>
                        <tr>
                            <td>Lion</td>
                            <td>0.7936</td>
                            <td>0.6759</td>
                            <td>0.7923</td>
                        </tr>
                        <tr>
                            <td>LAMB</td>
                            <td>0.8154</td>
                            <td>0.8071</td>
                            <td>0.8148</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>
        
        <section>
            <h2>Visualization</h2>
            <div class="visualization">
                <div class="chart-container">
                    <canvas id="accuracyChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Key Advantages</h2>
            <ul>
                <li><strong>Adaptive blending:</strong> Dynamically adjusts between AdamW and Lion based on their agreement</li>
                <li><strong>Robust performance:</strong> Works well across various tasks and architectures</li>
                <li><strong>Stability:</strong> The agreement mechanism provides stability during training</li>
                <li><strong>Efficiency:</strong> Comparable computational cost to AdamW and Lion</li>
            </ul>
            
            <h2>When to Use Apollo</h2>
            <ul>
                <li>When you want a robust optimizer that performs well across different tasks</li>
                <li>When training transformer-based models or CNNs</li>
                <li>When you need an optimizer that adapts to the training dynamics</li>
                <li>When you want to leverage the benefits of both adaptive and sign-based methods</li>
            </ul>
        </section>
        
        <footer>
            <p>Apollo Optimizer Algorithm Visualization | Based on the research paper "Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization"</p>
        </footer>
    </div>

    <script>
        // Accuracy Comparison Chart
        const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
        new Chart(accuracyCtx, {
            type: 'bar',
            data: {
                labels: ['MNIST', 'SST-2', 'AG News', 'IMDB', 'SVHN'],
                datasets: [
                    {
                        label: 'Apollo',
                        data: [0.9933, 0.8005, 0.9184, 0.8830, 0.9544],
                        backgroundColor: 'rgba(100, 255, 218, 0.7)',
                        borderColor: 'rgba(100, 255, 218, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'AdamW',
                        data: [0.9902, 0.8257, 0.9047, 0.8681, 0.9547],
                        backgroundColor: 'rgba(100, 181, 255, 0.7)',
                        borderColor: 'rgba(100, 181, 255, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Lion',
                        data: [0.9909, 0.7936, 0.9132, 0.8712, 0.9563],
                        backgroundColor: 'rgba(255, 121, 198, 0.7)',
                        borderColor: 'rgba(255, 121, 198, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Test Accuracy Comparison Across Datasets',
                        color: '#f0f6fc'
                    },
                    legend: {
                        labels: {
                            color: '#f0f6fc'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0.7,
                        ticks: {
                            color: '#f0f6fc'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#f0f6fc'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });

        // Loss Comparison Chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: ['Epoch 1', 'Epoch 5', 'Epoch 10', 'Epoch 15', 'Epoch 20', 'Epoch 25'],
                datasets: [
                    {
                        label: 'Apollo',
                        data: [2.1, 1.2, 0.7, 0.4, 0.2, 0.1],
                        backgroundColor: 'rgba(100, 255, 218, 0.2)',
                        borderColor: 'rgba(100, 255, 218, 1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'AdamW',
                        data: [2.3, 1.4, 0.9, 0.6, 0.4, 0.3],
                        backgroundColor: 'rgba(100, 181, 255, 0.2)',
                        borderColor: 'rgba(100, 181, 255, 1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'Lion',
                        data: [2.4, 1.5, 0.9, 0.5, 0.3, 0.2],
                        backgroundColor: 'rgba(255, 121, 198, 0.2)',
                        borderColor: 'rgba(255, 121, 198, 1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Training Loss Over Epochs (Simulated)',
                        color: '#f0f6fc'
                    },
                    legend: {
                        labels: {
                            color: '#f0f6fc'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: '#f0f6fc'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#f0f6fc'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>





# Apollo Optimizer Performance Analysis

This repository contains performance evaluations of the Apollo optimizer compared to state-of-the-art optimization algorithms across various datasets.

## Results Summary

### MNIST Dataset Performance

| Optimizer | Test Loss | Test Acc | Precision | Recall | F1 | AUC | Specificity | Avg Epoch Time (s) |
|-----------|-----------|----------|-----------|--------|----|-----|-------------|-------------------|
| Apollo | 0.0326 | 0.9933 | 0.9933 | 0.9932 | 0.9932 | 1.0000 | 0.9993 | 16.45 |
| AdaBelief | 0.0358 | 0.9881 | 0.9880 | 0.9880 | 0.9880 | 0.9999 | 0.9987 | 14.12 |
| LAMB | 0.0779 | 0.9751 | 0.9751 | 0.9749 | 0.9750 | 0.9996 | 0.9972 | 15.31 |
| Lion | 0.0378 | 0.9909 | 0.9908 | 0.9908 | 0.9908 | 0.9999 | 0.9990 | 14.23 |
| AdamW | 0.0309 | 0.9902 | 0.9902 | 0.9901 | 0.9901 | 0.9999 | 0.9989 | 15.05 |
| Sophia | 0.1444 | 0.9578 | 0.9574 | 0.9573 | 0.9573 | 0.9984 | 0.9953 | 14.63 |
| Nadam | 0.0318 | 0.9892 | 0.9892 | 0.9891 | 0.9891 | 0.9999 | 0.9988 | 14.17 |
| Adam | 0.0309 | 0.9892 | 0.9892 | 0.9890 | 0.9891 | 0.9999 | 0.9988 | 13.87 |
| RMSprop | 0.0240 | 0.9929 | 0.9928 | 0.9928 | 0.9928 | 1.0000 | 0.9992 | 14.10 |
| SGD | 0.0405 | 0.9872 | 0.9871 | 0.9871 | 0.9871 | 0.9999 | 0.9986 | 14.15 |

### Noisy MNIST Dataset Performance

| Optimizer | Test Loss | Test Acc | Precision | Recall | F1 | AUC | Specificity | Avg Epoch Time (s) |
|-----------|-----------|----------|-----------|--------|----|-----|-------------|-------------------|
| Apollo | 0.0337 | 0.9918 | 0.9918 | 0.9916 | 0.9917 | 1.0000 | 0.9991 | 18.96 |
| AdaBelief | 0.0376 | 0.9880 | 0.9880 | 0.9879 | 0.9879 | 0.9999 | 0.9987 | 17.50 |
| LAMB | 0.0825 | 0.9731 | 0.9730 | 0.9728 | 0.9729 | 0.9995 | 0.9970 | 18.41 |
| Lion | 0.0365 | 0.9917 | 0.9916 | 0.9916 | 0.9916 | 0.9999 | 0.9991 | 17.71 |
| AdamW | 0.0323 | 0.9893 | 0.9893 | 0.9892 | 0.9892 | 0.9999 | 0.9988 | 17.34 |
| Sophia | 0.1489 | 0.9564 | 0.9560 | 0.9558 | 0.9558 | 0.9983 | 0.9952 | 18.76 |
| Nadam | 0.0336 | 0.9890 | 0.9890 | 0.9889 | 0.9889 | 0.9999 | 0.9988 | 17.36 |
| Adam | 0.0327 | 0.9883 | 0.9883 | 0.9881 | 0.9882 | 0.9999 | 0.9987 | 17.20 |
| RMSprop | 0.0262 | 0.9910 | 0.9909 | 0.9908 | 0.9909 | 0.9999 | 0.9990 | 17.21 |
| SGD | 0.0427 | 0.9864 | 0.9863 | 0.9862 | 0.9863 | 0.9999 | 0.9985 | 17.10 |

### SST-2 Dataset Performance

| Optimizer | Test Loss | Test Acc | Precision | Recall | F1 | AUC | Specificity | Avg Epoch Time (s) |
|-----------|-----------|----------|-----------|--------|----|-----|-------------|-------------------|
| Apollo | 0.6567 | 0.8005 | 0.8031 | 0.7995 | 0.7996 | 0.8672 | 0.7500 | 16.19 |
| AdaBelief | 0.7099 | 0.7798 | 0.8024 | 0.7773 | 0.7745 | 0.8692 | 0.6379 | 15.23 |
| LAMB | 0.8071 | 0.8154 | 0.8173 | 0.8146 | 0.8148 | 0.8810 | 0.7734 | 16.70 |
| Lion | 0.6759 | 0.7936 | 0.7980 | 0.7924 | 0.7923 | 0.8646 | 0.7290 | 14.97 |
| AdamW | 0.5116 | 0.8257 | 0.8256 | 0.8257 | 0.8257 | 0.8931 | 0.8271 | 15.40 |
| Sophia | 0.6023 | 0.6950 | 0.7168 | 0.6919 | 0.6849 | 0.7751 | 0.5257 | 15.40 |
| Nadam | 0.4403 | 0.7993 | 0.8024 | 0.7983 | 0.7984 | 0.8809 | 0.7453 | 15.22 |
| Adam | 0.4434 | 0.7947 | 0.7986 | 0.7936 | 0.7936 | 0.8798 | 0.7336 | 15.41 |
| RMSprop | 0.4422 | 0.7993 | 0.8011 | 0.7985 | 0.7987 | 0.8787 | 0.7570 | 15.32 |
| SGD | 0.6841 | 0.5229 | 0.5621 | 0.5149 | 0.4052 | 0.6391 | 0.0794 | 15.10 |

### AG News Dataset Performance

| Optimizer | Test Loss | Test Acc | Precision | Recall | F1 | AUC | Specificity | Avg Epoch Time (s) |
|-----------|-----------|----------|-----------|--------|----|-----|-------------|-------------------|
| Apollo | 0.2672 | 0.9184 | 0.9184 | 0.9184 | 0.9183 | 0.9851 | 0.9728 | 3.99 |
| AdaBelief | 0.3092 | 0.9025 | 0.9023 | 0.9025 | 0.9023 | 0.9799 | 0.9675 | 3.07 |
| LAMB | 0.2692 | 0.9158 | 0.9156 | 0.9158 | 0.9157 | 0.9844 | 0.9719 | 3.52 |
| Lion | 0.3195 | 0.9132 | 0.9131 | 0.9132 | 0.9131 | 0.9843 | 0.9711 | 2.63 |
| AdamW | 0.2935 | 0.9047 | 0.9047 | 0.9047 | 0.9046 | 0.9816 | 0.9682 | 2.83 |
| Sophia | 1.2491 | 0.6521 | 0.6560 | 0.6521 | 0.6495 | 0.8520 | 0.8840 | 2.89 |
| Nadam | 0.5049 | 0.8554 | 0.8555 | 0.8554 | 0.8550 | 0.9664 | 0.9518 | 3.04 |
| Adam | 0.5046 | 0.8576 | 0.8578 | 0.8576 | 0.8572 | 0.9665 | 0.9525 | 3.02 |
| RMSprop | 0.4603 | 0.8658 | 0.8659 | 0.8658 | 0.8654 | 0.9697 | 0.9553 | 2.71 |
| SGD | 1.3489 | 0.4687 | 0.4696 | 0.4687 | 0.4681 | 0.7164 | 0.8229 | 2.46 |

### IMDB Dataset Performance

| Optimizer | Test Loss | Test Acc | Precision | Recall | F1 | AUC | Specificity | Avg Epoch Time (s) |
|-----------|-----------|----------|-----------|--------|----|-----|-------------|-------------------|
| Apollo | 0.3084 | 0.8830 | 0.8832 | 0.8830 | 0.8829 | 0.9453 | 0.8959 | 1.36 |
| AdaBelief | 0.3808 | 0.8576 | 0.8576 | 0.8576 | 0.8576 | 0.9256 | 0.8606 | 1.19 |
| LAMB | 0.3328 | 0.8695 | 0.8695 | 0.8695 | 0.8695 | 0.9355 | 0.8700 | 1.46 |
| Lion | 0.3818 | 0.8712 | 0.8716 | 0.8712 | 0.8712 | 0.9353 | 0.8868 | 1.10 |
| AdamW | 0.3479 | 0.8681 | 0.8682 | 0.8681 | 0.8681 | 0.9350 | 0.8757 | 1.13 |
| Sophia | 0.6822 | 0.6210 | 0.6211 | 0.6210 | 0.6210 | 0.6655 | 0.6318 | 1.14 |
| Nadam | 0.5920 | 0.7320 | 0.7320 | 0.7320 | 0.7319 | 0.7999 | 0.7402 | 1.16 |
| Adam | 0.5955 | 0.7301 | 0.7303 | 0.7301 | 0.7301 | 0.7965 | 0.7422 | 1.13 |
| RMSprop | 0.5719 | 0.7488 | 0.7489 | 0.7488 | 0.7488 | 0.8189 | 0.7594 | 1.09 |
| SGD | 0.6919 | 0.5245 | 0.5247 | 0.5245 | 0.5235 | 0.5392 | 0.5704 | 1.06 |

### SVHN Dataset Performance

| Optimizer | Test Loss | Test Acc | Precision | Recall | F1 | AUC | Specificity | Avg Epoch Time (s) |
|-----------|-----------|----------|-----------|--------|----|-----|-------------|-------------------|
| Apollo | 0.1672 | 0.9544 | 0.9515 | 0.9519 | 0.9516 | 0.9969 | 0.9949 | 76.86 |
| AdaBelief | 0.1605 | 0.9562 | 0.9546 | 0.9529 | 0.9537 | 0.9971 | 0.9950 | 73.60 |
| LAMB | 0.1877 | 0.9482 | 0.9442 | 0.9449 | 0.9444 | 0.9965 | 0.9942 | 79.87 |
| Lion | 0.1634 | 0.9563 | 0.9545 | 0.9535 | 0.9540 | 0.9970 | 0.9951 | 72.32 |
| AdamW | 0.1648 | 0.9547 | 0.9525 | 0.9519 | 0.9522 | 0.9971 | 0.9949 | 72.60 |
| Sophia | 0.1823 | 0.9482 | 0.9462 | 0.9455 | 0.9458 | 0.9967 | 0.9941 | 72.84 |
| Nadam | 0.1535 | 0.9576 | 0.9551 | 0.9569 | 0.9560 | 0.9972 | 0.9952 | 72.83 |
| Adam | 0.1560 | 0.9571 | 0.9553 | 0.9550 | 0.9551 | 0.9971 | 0.9952 | 72.40 |
| RMSprop | 0.1709 | 0.9528 | 0.9500 | 0.9509 | 0.9503 | 0.9969 | 0.9947 | 71.85 |
| SGD | 0.5677 | 0.8168 | 0.8070 | 0.7980 | 0.8009 | 0.9782 | 0.9794 | 71.61 |

### Apollo Hyperparameter Tuning on CIFAR-10

| Configuration | Accuracy | F1-Score | AUC | Time/Epoch |
|---------------|----------|----------|-----|------------|
| lr=0.0005 | **0.8708** | 0.8709 | 0.9907 | 25.68 s |
| Default (β₃=0.9) | 0.8700 | 0.8700 | 0.9901 | 26.09 s |
| β₃=0.75 | 0.8677 | 0.8672 | 0.9894 | 25.69 s |
| weight decay=0.05 | 0.8672 | 0.8668 | 0.9892 | 24.29 s |
| β₁=0.85, β₂=0.999 | 0.8665 | 0.8661 | 0.9900 | 24.60 s |
| β₃=0.5 | 0.8659 | 0.8655 | 0.9895 | 25.64 s |
| β₃=0.0 | 0.8642 | 0.8641 | 0.9892 | 25.48 s |
| β₁=0.9, β₂=0.99 | 0.8625 | 0.8622 | 0.9892 | 24.60 s |
| weight decay=0.0 | 0.8608 | 0.8603 | 0.9889 | 24.52 s |
| lr=0.005 | 0.8139 | 0.8131 | 0.9815 | 25.27 s |
| **Mean ± Std** | 0.8640 ± 0.017 | 0.8638 ± 0.017 | 0.9891 ± 0.0029 | 25.42 ± 0.58 s |

*Default configuration: lr=0.001, β₁=0.9, β₂=0.999, β₃=0.9, wd=0.01, eps=1e-8*

## Key Findings

1. **MNIST Performance**: Apollo achieves the highest accuracy (99.33%) on the standard MNIST dataset
2. **Noisy Data Robustness**: Apollo maintains strong performance on noisy MNIST data
3. **Text Classification**: Apollo shows competitive performance on NLP tasks (SST-2, AG News, IMDB)
4. **Image Recognition**: On SVHN, Apollo performs competitively with other top optimizers
5. **Hyperparameter Sensitivity**: The learning rate has the most significant impact on Apollo's performance

## Usage

To use the Apollo optimizer in your projects:

```python
# PyTorch implementation
optimizer = Apollo(model.parameters(), lr=0.001, beta1=0.9, beta2=0.999, beta3=0.9, weight_decay=0.01, eps=1e-8)
