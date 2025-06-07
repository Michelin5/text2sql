// app.js

// DOM Element References (keep them organized)
const DOMElements = {
    processButton: document.getElementById('processButton'),
    buttonSpinner: document.getElementById('buttonSpinner'),
    buttonText: document.getElementById('buttonText'),
    errorMessageDiv: document.getElementById('errorMessage'),
    errorText: document.getElementById('errorText'),
    coordinationSection: document.getElementById('coordinationSection'),
    userPromptInput: document.getElementById('userPrompt'),
    clarificationOutput: document.getElementById('clarificationOutput'),
    stepAgent1: document.getElementById('step-agent1'),
    stepAgent2: document.getElementById('step-agent2'),
    stepAgent3: document.getElementById('step-agent3'),
    stepAgent4: document.getElementById('step-agent4'),
    stepAnomaly: document.getElementById('step-anomaly'),
};

// Output elements for dynamic content
const outputElements = {
    coordination: document.getElementById('coordinationOutput'),
    queryEnhancement: document.getElementById('queryEnhancementOutput'),
    clarification: document.getElementById('clarificationOutput'),
    rephrasedPrompt: document.getElementById('rephrasedPromptOutput'),
    plan: document.getElementById('planOutput'),
    sql: document.getElementById('sqlOutput'),
    sqlValidation: document.getElementById('sqlValidationOutput'),
    execution: document.getElementById('executionOutput'),
    sqlError: document.getElementById('sqlErrorOutput'),
    anomaly: document.getElementById('anomalyOutput'),
    finalAnswer: document.getElementById('finalAnswerOutput')
};

const BACKEND_API_URL = 'http://127.0.0.1:5002/process_query_stream';

let eventSource = null; // Initialize as null

// --- Helper Functions ---

/**
 * Sets the loading state of the process button.
 * @param {boolean} isLoading - True to show loading state, false otherwise.
 */
function setButtonLoading(isLoading) {
    if (DOMElements.processButton) {
        DOMElements.processButton.disabled = isLoading;
        if (isLoading) {
            DOMElements.buttonSpinner?.classList.remove('hidden');
            DOMElements.buttonText.textContent = 'Обработка...';
        } else {
            DOMElements.buttonSpinner?.classList.add('hidden');
            DOMElements.buttonText.textContent = 'Анализировать';
        }
    }
}

/**
 * Displays an error message to the user.
 * @param {string} message - The error message to display.
 */
function showError(message) {
    DOMElements.errorMessageDiv?.classList.remove('hidden');
    DOMElements.errorText.textContent = message;
    setButtonLoading(false);
}

/**
 * Hides the error message.
 */
function hideError() {
    DOMElements.errorMessageDiv?.classList.add('hidden');
    DOMElements.errorText.textContent = '';
}

/**
 * Resets the entire interface to its initial state.
 */
function resetInterface() {
    hideError();
    DOMElements.coordinationSection?.classList.add('hidden');

    Object.values(outputElements).forEach(element => {
        if (element) {
            element.innerHTML = ''; // Очищаем содержимое
        }
    });

    // Reset final answer to its initial placeholder state
    if (outputElements.finalAnswer) {
        outputElements.finalAnswer.innerHTML = `
            <div class="text-center py-8">
                <svg class="w-12 h-12 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                </svg>
                <p class="text-gray-500">Здесь появится результат анализа</p>
            </div>
        `;
    }

    // Reset all step items to pending and default spinner color
    document.querySelectorAll('.step-item').forEach(step => {
        step.classList.remove('step-completed', 'step-active', 'step-error');
        step.classList.add('step-pending');
        const spinner = step.querySelector('.w-2.h-2');
        if (spinner) {
            spinner.classList.remove('bg-green-500', 'bg-blue-500', 'bg-red-500', 'animate-pulse');
            spinner.classList.add('bg-gray-400');
        }
    });

    // Ensure hidden state for specific dynamic sections
    DOMElements.stepAnomaly?.classList.add('hidden');
    DOMElements.sqlErrorOutput?.classList.add('hidden');
    if (DOMElements.sqlErrorOutput) {
        DOMElements.sqlErrorOutput.querySelector('div').textContent = '';
    }

    // Close any existing EventSource connection
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
}

/**
 * Updates the visual status of a pipeline step.
 * @param {string} stepId - The ID of the step element (e.g., 'step-agent1').
 * @param {'active'|'completed'|'error'|'pending'} status - The new status.
 */
function updateStep(stepId, status) {
    const stepElement = document.getElementById(stepId);
    if (!stepElement) {
        console.warn(`Step element with ID "${stepId}" not found.`);
        return;
    }

    stepElement.classList.remove('step-completed', 'step-active', 'step-pending', 'step-error');
    const spinner = stepElement.querySelector('.w-2.h-2');

    switch (status) {
        case 'active':
            stepElement.classList.add('step-active');
            if (spinner) {
                spinner.classList.remove('bg-gray-400', 'bg-green-500', 'bg-red-500');
                spinner.classList.add('bg-blue-500', 'animate-pulse');
            }
            break;
        case 'completed':
            stepElement.classList.add('step-completed');
            if (spinner) {
                spinner.classList.remove('bg-gray-400', 'bg-blue-500', 'animate-pulse', 'bg-red-500');
                spinner.classList.add('bg-green-500');
            }
            break;
        case 'error':
            stepElement.classList.add('step-error');
            if (spinner) {
                spinner.classList.remove('bg-gray-400', 'bg-blue-500', 'animate-pulse', 'bg-green-500');
                spinner.classList.add('bg-red-500');
            }
            break;
        case 'pending':
        default:
            stepElement.classList.add('step-pending');
            if (spinner) {
                spinner.classList.remove('bg-green-500', 'bg-blue-500', 'bg-red-500', 'animate-pulse');
                spinner.classList.add('bg-gray-400');
            }
            break;
    }
}

/**
 * Initiates the query processing pipeline via Server-Sent Events.
 */
function runPipeline() {
    const userPrompt = DOMElements.userPromptInput.value.trim();
    if (!userPrompt) {
        showError("Пожалуйста, введите ваш запрос для анализа данных.");
        return;
    }

    resetInterface();
    setButtonLoading(true);

    const streamUrl = `${BACKEND_API_URL}?user_prompt=${encodeURIComponent(userPrompt)}`;

    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    eventSource = new EventSource(streamUrl);

    eventSource.onopen = function() {
        console.log("Connection to server opened.");
        hideError();
    };

    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);

            if (data.step === 'done') {
                console.log("Stream finished.");
                eventSource.close();
                eventSource = null;
                setButtonLoading(false);
                return;
            }

            updateUI(data.step, data.content);
        } catch (e) {
            console.error("Error parsing EventSource message:", e, event.data);
            showError("Ошибка обработки данных от сервера.");
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            setButtonLoading(false);
        }
    };

    eventSource.onerror = function(err) {
        console.error("EventSource failed:", err);
        showError("Ошибка соединения с сервером. Проверьте подключение или логи сервера.");
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        setButtonLoading(false);
    };
}

/**
 * Updates the user interface based on incoming SSE data.
 * @param {string} step - The name of the pipeline step.
 * @param {any} content - The data associated with the step.
 */
function updateUI(step, content) {
    switch (step) {
        case 'coordination':
            DOMElements.coordinationSection?.classList.remove('hidden');
            displayCoordination(content);
            break;

        case 'query_enhanced':
            displayQueryEnhancement(content);
            break;

        case 'clarification_needed':
            displayClarificationRequest(content);
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            setButtonLoading(false);
            // Hide process steps when clarification is needed
            DOMElements.stepAgent1?.classList.add('hidden');
            DOMElements.stepAgent2?.classList.add('hidden');
            DOMElements.stepAgent3?.classList.add('hidden');
            DOMElements.stepAgent4?.classList.add('hidden');
            DOMElements.stepAnomaly?.classList.add('hidden');
            break;

        case 'formal_request':
            updateStep('step-agent1', 'completed');
            outputElements.rephrasedPrompt.innerHTML = `<div class="text-sm">${content || "Нет данных."}</div>`;
            updateStep('step-agent2', 'active');
            break;

        case 'plan':
            updateStep('step-agent2', 'completed');
            outputElements.plan.innerHTML = `<div class="text-sm whitespace-pre-line">${content || "Нет данных."}</div>`;
            updateStep('step-agent3', 'active');
            break;

        case 'generated_sql_query':
            outputElements.sql.innerHTML = `<pre class="sql-block text-xs p-3 rounded-lg overflow-x-auto">${content || "-- SQL-запрос не сгенерирован --"}</pre>`;
            break;

        case 'executed_sql_query':
            updateStep('step-agent3', 'completed');
            updateStep('step-agent4', 'active');
            outputElements.execution.innerHTML = `
                <div class="flex items-center space-x-2 text-sm text-blue-600">
                    <div class="loading-spinner"></div>
                    <span>Выполнение SQL-запроса...</span>
                </div>
            `;
            break;

        case 'sql_results_str':
            updateStep('step-agent4', 'completed');
            outputElements.execution.innerHTML = `
                <div class="flex items-center space-x-2 text-sm text-green-600">
                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                    </svg>
                    <span>Данные успешно получены</span>
                </div>
            `;
            break;

        case 'sql_error':
            updateStep('step-agent4', 'error');
            const errorOutput = outputElements.sqlError;
            if (errorOutput) {
                errorOutput.querySelector('div').textContent = content || "Неизвестная ошибка SQL.";
                errorOutput.classList.remove('hidden');
            }
            outputElements.execution.innerHTML = `
                <div class="flex items-center space-x-2 text-sm text-red-600">
                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                    </svg>
                    <span>Ошибка выполнения SQL</span>
                </div>
            `;
            break;

        case 'sql_validation_log':
            outputElements.sqlValidation.innerHTML += createValidationLogCard(content);
            break;

        case 'anomaly_analysis':
            DOMElements.stepAnomaly?.classList.remove('hidden');
            updateStep('step-anomaly', 'active');
            displayAnomalyResults(content);
            break;

        case 'final_answer':
            if (typeof marked !== 'undefined' && outputElements.finalAnswer) {
                updateStep('step-anomaly', 'completed');
                outputElements.finalAnswer.innerHTML = marked.parse(content || "<p>Ответ не был сгенерирован.</p>", { breaks: true, gfm: true });
            } else if (outputElements.finalAnswer) {
                outputElements.finalAnswer.innerHTML = `<p>${content || "Ответ не был сгенерирован."}</p>`;
                console.warn("Marked.js library not found. Final answer displayed as plain text.");
            }
            break;

        case 'error':
            showError(content || "Произошла неизвестная ошибка.");
            break;

        default:
            console.warn("Получен неизвестный шаг:", step, content);
    }
}

// --- Dynamic Content Rendering Functions ---

function createValidationLogCard(logData) {
    if (!logData) return '';
    return `
        <div class="mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div class="text-xs text-yellow-800">
                <strong>Попытка исправления ${logData.fix_attempt || 'N/A'}:</strong>
                ${logData.error || 'Нет данных об ошибке.'}
            </div>
        </div>
    `;
}

function displayCoordination(coordinationData) {
    const element = outputElements.coordination;
    if (!element) return;
    if (!coordinationData) {
        element.innerHTML = '<div class="text-sm text-gray-600">Нет данных координации.</div>';
        return;
    }
    const needsAnomaly = coordinationData.needs_anomaly_detection;
    const analysisType = coordinationData.analysis_type;
    const keywords = coordinationData.keywords || [];

    element.innerHTML = `
        <div class="bg-white rounded-xl card-shadow p-6 animate-fade-in">
            <div class="flex items-center space-x-3 mb-4">
                <div class="w-10 h-10 gradient-bg rounded-lg flex items-center justify-center">
                    <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                    </svg>
                </div>
                <div>
                    <h3 class="text-lg font-semibold text-gray-900">Координация запроса</h3>
                    <p class="text-sm text-gray-600">Анализ типа обработки</p>
                </div>
            </div>
            <div class="flex flex-wrap gap-2">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${needsAnomaly ? 'bg-orange-100 text-orange-800' : 'bg-blue-100 text-blue-800'}">
                    ${analysisType === 'anomaly' ? '🔍 Поиск аномалий' : '📊 Стандартный анализ'}
                </span>
                ${needsAnomaly ? '<span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">🚨 Агент аномалий активирован</span>' : ''}
            </div>
            ${keywords.length > 0 ? `<div class="mt-3 text-sm text-gray-600"><strong>Ключевые слова:</strong> ${keywords.join(', ')}</div>` : ''}
        </div>
    `;
}

function displayQueryEnhancement(enhancementData) {
    const element = outputElements.queryEnhancement;
    if (!element) return;
    if (!enhancementData) {
        element.innerHTML = '<div class="text-sm text-gray-600">Нет данных об улучшении запроса.</div>';
        return;
    }
    element.innerHTML = `
        <div class="bg-white rounded-xl card-shadow p-6 animate-fade-in">
            <div class="flex items-center space-x-3 mb-4">
                <div class="w-10 h-10 gradient-warning rounded-lg flex items-center justify-center">
                    <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                    </svg>
                </div>
                <div>
                    <h3 class="text-lg font-semibold text-gray-900">Улучшение запроса</h3>
                    <p class="text-sm text-gray-600">Автоматическая оптимизация</p>
                </div>
            </div>
            <div class="space-y-3">
                <div>
                    <span class="text-xs font-medium text-gray-500 uppercase tracking-wide">Исходный запрос</span>
                    <p class="text-sm text-gray-700 mt-1">${enhancementData.original_prompt || "Нет данных."}</p>
                </div>
                <div>
                    <span class="text-xs font-medium text-gray-500 uppercase tracking-wide">Улучшенный запрос</span>
                    <p class="text-sm text-gray-900 font-medium mt-1">${enhancementData.enhanced_prompt || "Нет данных."}</p>
                </div>
                <div class="text-xs text-gray-600 bg-gray-50 p-3 rounded-lg">
                    ${enhancementData.reason || "Нет данных о причине."}
                </div>
            </div>
        </div>
    `;
}

function displayClarificationRequest(clarificationData) {
    const element = outputElements.clarification;
    if (!element) return;
    if (!clarificationData) {
        element.innerHTML = '<div class="text-sm text-gray-600">Нет данных для уточнения.</div>';
        return;
    }

    let suggestionsHTML = '';
    if (clarificationData.suggested_tables && clarificationData.suggested_tables.length > 0) {
        suggestionsHTML = `
            <div class="mt-6">
                <h4 class="text-sm font-medium text-gray-900 mb-3">Выберите тип данных:</h4>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    ${clarificationData.suggested_tables.map(table =>
                        `<button type="button" onclick="window.selectTable('${table.id}', '${table.name}')" class="p-4 text-left border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-all duration-200 group">
                            <div class="font-medium text-gray-900 group-hover:text-blue-700">${table.name}</div>
                            <div class="text-xs text-gray-500 mt-1">${table.description}</div>
                        </button>`
                    ).join('')}
                </div>
            </div>
        `;
    }

    element.innerHTML = `
        <div class="bg-white rounded-xl card-shadow p-6 animate-fade-in">
            <div class="flex items-center space-x-3 mb-4">
                <div class="w-10 h-10 gradient-info rounded-lg flex items-center justify-center">
                    <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                </div>
                <div>
                    <h3 class="text-lg font-semibold text-gray-900">Требуется уточнение</h3>
                    <p class="text-sm text-gray-600">Для более точного анализа</p>
                </div>
            </div>
            <p class="text-gray-700 mb-4">${clarificationData.message || 'Пожалуйста, уточните ваш запрос'}</p>
            ${suggestionsHTML}
            <div class="mt-6">
                <h4 class="text-sm font-medium text-gray-900 mb-3">Или переформулируйте запрос:</h4>
                <textarea
                    id="clarificationPrompt"
                    rows="2"
                    class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all duration-200"
                    placeholder="Например: Уточните, что интересуют данные о населении из таблицы 'demography'"
                ></textarea>
                <button
                    type="button"
                    onclick="window.sendClarification()"
                    class="mt-3 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-all duration-200"
                >
                    Отправить уточнение
                </button>
            </div>
        </div>
    `;
    DOMElements.coordinationSection?.classList.remove('hidden');
}

function displayAnomalyResults(anomalyData) {
    const element = outputElements.anomaly;
    if (!element) return;
    if (!anomalyData) {
        element.innerHTML = '<div class="text-sm text-gray-600">Нет данных анализа аномалий.</div>';
        updateStep('step-anomaly', 'pending');
        return;
    }

    let content = '';
    if (anomalyData.status === 'success' && anomalyData.anomalies && anomalyData.anomalies.length > 0) {
        content = `
            <div class="anomaly-results space-y-4">
                <p class="text-green-700 text-sm font-semibold flex items-center">
                    <svg class="w-5 h-5 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                    </svg>
                    Обнаружены аномалии:
                </p>
                <ul class="list-disc list-inside text-sm text-gray-800">
                    ${anomalyData.anomalies.map(anomaly => `<li>${anomaly}</li>`).join('')}
                </ul>
            </div>
        `;
        updateStep('step-anomaly', 'completed');
    } else if (anomalyData.status === 'success' && anomalyData.anomalies && anomalyData.anomalies.length === 0) {
        content = `
            <p class="text-green-700 text-sm font-semibold flex items-center">
                <svg class="w-5 h-5 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                </svg>
                Аномалий не обнаружено.
            </p>
        `;
        updateStep('step-anomaly', 'completed');
    } else if (anomalyData.status === 'error') {
        content = `
            <p class="text-red-700 text-sm font-semibold flex items-center">
                <svg class="w-5 h-5 mr-2 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                </svg>
                Ошибка при анализе аномалий: ${anomalyData.error || 'Неизвестная ошибка.'}
            </p>
        `;
        updateStep('step-anomaly', 'error');
    } else {
        content = `<div class="text-sm text-gray-600">Ожидание результатов анализа аномалий...</div>`;
        updateStep('step-anomaly', 'active');
    }
    element.innerHTML = content;
}

// --- Global Functions (exposed for onclick attributes in dynamically generated HTML) ---

window.selectTable = function(tableId, tableName) {
    const userPrompt = DOMElements.userPromptInput.value;
    const clarificationPrompt = `Пожалуйста, используй данные из таблицы '${tableName}' (ID: ${tableId}) для запроса: ${userPrompt}`;
    DOMElements.userPromptInput.value = clarificationPrompt;
    DOMElements.clarificationOutput.innerHTML = '';
    DOMElements.stepAgent1?.classList.remove('hidden');
    DOMElements.stepAgent2?.classList.remove('hidden');
    DOMElements.stepAgent3?.classList.remove('hidden');
    DOMElements.stepAgent4?.classList.remove('hidden');
    DOMElements.stepAnomaly?.classList.remove('hidden');
    runPipeline();
}

window.sendClarification = function() {
    const clarificationPromptEl = document.getElementById('clarificationPrompt');
    const clarificationText = clarificationPromptEl ? clarificationPromptEl.value.trim() : '';

    if (clarificationText) {
        DOMElements.userPromptInput.value = clarificationText;
        DOMElements.clarificationOutput.innerHTML = '';
        DOMElements.stepAgent1?.classList.remove('hidden');
        DOMElements.stepAgent2?.classList.remove('hidden');
        DOMElements.stepAgent3?.classList.remove('hidden');
        DOMElements.stepAgent4?.classList.remove('hidden');
        DOMElements.stepAnomaly?.classList.remove('hidden');
        runPipeline();
    } else {
        showError("Пожалуйста, введите уточнение или выберите один из вариантов.");
    }
}

window.useExample = function(exampleText) {
    DOMElements.userPromptInput.value = exampleText;
    runPipeline();
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    if (DOMElements.processButton) {
        DOMElements.processButton.addEventListener('click', runPipeline);
    } else {
        console.error("Error: processButton element not found. Make sure an element with id='processButton' exists in your HTML.");
    }

    if (DOMElements.userPromptInput) {
        DOMElements.userPromptInput.addEventListener('keydown', (e) => {
            if (e.shiftKey && e.key === 'Enter') {
                e.preventDefault();
                runPipeline();
            }
        });
    } else {
        console.error("Error: userPrompt element not found.");
    }

    resetInterface();
});