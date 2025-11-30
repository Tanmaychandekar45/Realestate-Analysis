import React, { useState, useCallback } from 'react';
// Using fetch directly with exponential backoff is preferred for robustness over axios
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Search, Loader2, TrendingUp, DollarSign, MapPin, Layers, Target, BarChart3, Rocket, XCircle, Download } from 'lucide-react';

// Base URL for the Django API. 
// Note: This must match your Django setup (e.g., if Django is running on port 8000, use that URL)
const API_BASE_URL = 'http://127.0.0.1:8000'; 
const ANALYZE_URL = `${API_BASE_URL}/api/analyze/`;
const DOWNLOAD_URL = `${API_BASE_URL}/api/download/`;

// --- Utility Components ---

const MetricDisplay = ({ icon: Icon, title, value, colorClass, borderColorClass }) => (
    <div className={`p-4 bg-gray-900 rounded-lg shadow-xl border ${borderColorClass} transition duration-300 hover:shadow-cyan-500/30 hover:scale-[1.02] transform`}>
        <div className={`flex items-center justify-between mb-2`}>
            <p className="text-sm font-mono uppercase text-gray-400">{title}</p>
            <Icon className={`w-6 h-6 ${colorClass}`} />
        </div>
        <p className="text-3xl font-extrabold text-white overflow-hidden text-ellipsis whitespace-nowrap">{value}</p>
    </div>
);

const LoadingConsole = () => (
    <div className="flex flex-col items-center justify-center p-12 bg-gray-900 rounded-xl shadow-2xl border border-cyan-500/30 animate-pulse-slow">
        <Loader2 className="w-16 h-16 text-cyan-400 animate-spin" />
        <p className="mt-6 text-2xl font-bold text-cyan-400">EXECUTING ANALYSIS QUERY...</p>
        <p className="text-base text-gray-500 mt-2 font-mono">STAND BY: Data stream decryption in progress...</p>
    </div>
);

// --- Main Application Logic ---

const App = () => {
    const [query, setQuery] = useState('');
    const [analysisResult, setAnalysisResult] = useState(null); 
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [lastSuccessfulQuery, setLastSuccessfulQuery] = useState('');

    const handleAnalyze = async (e) => {
        e.preventDefault(); 
        if (!query.trim()) return;

        setLoading(true);
        setError(null);
        setAnalysisResult(null);

        const maxRetries = 3;
        let delay = 1000; 
        
        for (let i = 0; i < maxRetries; i++) {
            try {
                const response = await fetch(ANALYZE_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query }),
                });

                if (response.status === 404) {
                    throw new Error("404 Not Found. Check Django's URL configuration and the API_BASE_URL.");
                }

                const data = await response.json();

                if (response.ok) {
                    setAnalysisResult(data);
                    setLastSuccessfulQuery(query);
                    setLoading(false);
                    return; // Success
                } else {
                    throw new Error(data.error || `Request failed with status ${response.status}.`);
                }
            } catch (err) {
                if (i < maxRetries - 1) {
                    await new Promise(resolve => setTimeout(resolve, delay));
                    delay *= 2; 
                } else {
                    console.error("API call failed after multiple retries:", err);
                    setError(`Error connecting to analysis engine: ${err.message}.`);
                    setLoading(false);
                }
            }
        }
    };
    
    // ====================================================================
    // DOWNLOAD FUNCTIONALITY 
    // ====================================================================
    const handleDownload = async () => {
        const tableDataLength = analysisResult?.tableData?.length || 0;
        
        if (!lastSuccessfulQuery || tableDataLength === 0) {
            setError("Please run a successful query before attempting to download data.");
            return;
        }
        
        try {
            const response = await fetch(DOWNLOAD_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: lastSuccessfulQuery }),
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Download request failed: ${response.status} - ${errorText}`);
            }

            // Get the Blob data
            const blob = await response.blob(); 
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            
            let filename = 'real_estate_data.csv'; 

            // Attempt to get filename from Content-Disposition header
            const contentDisposition = response.headers.get('Content-Disposition');
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="(.+)"/);
                if (filenameMatch && filenameMatch.length === 2) {
                    filename = filenameMatch[1].replace(/['"]/g, ''); 
                }
            }
            
            // Trigger the download
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url); 
            
        } catch (err) {
            console.error("Download failed:", err);
            setError(`Failed to download data: ${err.message}`);
        }
    };
    // ====================================================================


    const chartData = analysisResult?.chartData?.data || [];
    const tableData = analysisResult?.tableData || [];
    const summaryText = analysisResult?.summary || "Analysis Terminal Ready. Input your market query.";

    // Derived Statistics
    const totalProperties = tableData.length;
    // Calculate avgPrice across all data points in the chart, using 'price' key from backend
    const avgPrice = chartData.length > 0 
        ? chartData.reduce((sum, item) => sum + (item['price'] || 0), 0) / chartData.length
        : null;
    // Get the latest demand from the last data point, using 'demand' key from backend
    const latestDemand = chartData.length > 0 
        ? chartData[chartData.length - 1]['demand']
        : null;
        
    // Formatters
    const getFormattedPrice = (price) => (price !== null && !isNaN(price)) ? `₹${price.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}` : 'N/A';
    const getFormattedDemand = (demand) => (demand !== null && !isNaN(demand)) ? `${demand.toFixed(0)} units` : 'N/A';
    // Format chart price ticks to be cleaner (e.g., '₹50k')
    const getFormattedChartPrice = (price) => `₹${(price / 1000).toFixed(0)}k`;


    const ChartComponent = () => (
        <div className="p-6 bg-gray-900 rounded-xl shadow-2xl border border-indigo-500/50">
            <h3 className="text-xl font-bold text-indigo-400 mb-4 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-lime-400" />
                Quantum Price & Demand Projection
            </h3>
            {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={chartData} margin={{ top: 15, right: 30, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="5 5" stroke="#374151" />
                        {/* dataKey fixed to lowercase 'year' */}
                        <XAxis dataKey="year" tickLine={false} axisLine={{ stroke: '#4b5563' }} stroke="#9ca3af" /> 
                        <YAxis yAxisId="left" stroke="#3b82f6" tickFormatter={getFormattedChartPrice} tick={{ fill: '#93c5fd' }} /> 
                        <YAxis yAxisId="right" orientation="right" stroke="#10b981" tick={{ fill: '#34d399' }} /> 
                        <Tooltip 
                            contentStyle={{ borderRadius: '8px', border: '1px solid #1f2937', backgroundColor: '#1f2937', padding: '10px' }}
                            labelStyle={{ color: '#fff' }}
                            formatter={(value, name, props) => {
                                // Determine the key based on the dataKey
                                const key = props.dataKey === 'price' ? 'Average Price' : 'Average Demand';
                                const formattedValue = props.dataKey === 'price' 
                                    ? getFormattedPrice(value) 
                                    : getFormattedDemand(value);
                                return [formattedValue, key];
                            }}
                            labelFormatter={(label) => `Year: ${label}`}
                        />
                        <Legend wrapperStyle={{ paddingTop: '15px', color: '#fff' }} />
                        {/* dataKey fixed to lowercase 'price' and 'demand' */}
                        <Line yAxisId="left" type="monotone" dataKey="price" name="Average Price (₹/sqft)" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4, fill: '#3b82f6' }} activeDot={{ r: 8, stroke: '#3b82f6', strokeWidth: 2 }} />
                        <Line yAxisId="right" type="monotone" dataKey="demand" name="Average Demand (Units Sold)" stroke="#10b981" strokeWidth={3} dot={{ r: 4, fill: '#10b981' }} activeDot={{ r: 8, stroke: '#10b981', strokeWidth: 2 }} />
                    </LineChart>
                </ResponsiveContainer>
            ) : (
                <div className="flex items-center justify-center h-[400px] text-gray-500">
                    <p>Awaiting high-fidelity data stream from backend servers.</p>
                </div>
            )}
        </div>
    );

    const TableComponent = () => (
        <div className="p-6 bg-gray-900 rounded-xl shadow-2xl border border-indigo-500/50" style={{ overflowX: 'auto' }}>
            <h3 className="text-xl font-bold text-indigo-400 mb-4 flex items-center">
                <Layers className="w-5 h-5 mr-2 text-cyan-400" />
                Raw Data Matrix ({tableData.length} records processed)
            </h3>
            
            {/* DOWNLOAD BUTTON - Placed above the table for easy access */}
            <div className="flex justify-end mb-4">
                <button 
                    onClick={handleDownload}
                    disabled={tableData.length === 0}
                    className={`flex items-center px-4 py-2 text-sm font-semibold rounded-lg transition duration-200 
                        ${tableData.length > 0 && lastSuccessfulQuery
                            ? 'bg-cyan-600 text-white hover:bg-cyan-500 shadow-md hover:shadow-cyan-500/50' 
                            : 'bg-gray-700 text-gray-400 cursor-not-allowed'
                        }`}
                >
                    <Download className="w-4 h-4 mr-2" />
                    Download CSV ({tableData.length} records)
                </button>
            </div>
            {/* END DOWNLOAD BUTTON */}

            <div className="max-h-[500px] overflow-y-auto border border-gray-700 rounded-lg">
                {tableData.length > 0 ? (
                    <table className="min-w-full border-collapse">
                        <thead className="bg-gray-800 sticky top-0 shadow-md">
                            <tr>
                                {Object.keys(tableData[0]).map(key => (
                                    <th key={key} className="p-3 text-left text-xs font-mono font-semibold text-cyan-400 uppercase whitespace-nowrap">
                                        {key.replace(/_/g, ' ')}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="bg-gray-900 divide-y divide-gray-800">
                            {tableData.map((row, index) => (
                                <tr key={index} className="hover:bg-gray-800/50 transition-colors duration-150">
                                    {Object.values(row).map((value, idx) => (
                                        <td key={idx} className="p-3 whitespace-nowrap text-sm font-mono text-gray-300">
                                            {typeof value === 'number' ? value.toLocaleString('en-IN', { maximumFractionDigits: 2 }) : value || '-'}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <p className="p-8 text-gray-500 font-mono">Raw data stream currently empty.</p>
                )}
            </div>
        </div>
    );


    return (
        // Your main structure here
        <div className="min-h-screen bg-gray-900 p-8 md:p-16">
            <header className="mb-10 text-center">
                <h1 className="shadow-text text-5xl font-extrabold text-cyan-400 tracking-wider">
                    <span className="text-indigo-400">PROJECT ATLAS:</span> MARKET ANALYSIS CONSOLE
                </h1>
                <p className="text-lg text-gray-500 mt-3 font-mono">Query the real estate matrix for predictive insights.</p>
            </header>

            {/* Query Input Form - The Command Line */}
            <div className="max-w-4xl mx-auto mb-10">
                <form onSubmit={handleAnalyze} className="flex flex-col md:flex-row gap-4 p-4 bg-gray-800 rounded-xl shadow-2xl border border-indigo-500/50">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="[SYSTEM_QUERY] > Initiate analysis on: 'Wakad' or 'Baner'..."
                        className="flex-grow p-4 text-lg font-mono text-cyan-400 bg-gray-950 border-2 border-indigo-600 rounded-lg shadow-inner outline-none focus:border-cyan-400 transition"
                        disabled={loading}
                    />
                    <button
                        type="submit"
                        className="flex items-center justify-center px-8 py-3 text-xl font-bold text-white bg-indigo-600 rounded-lg shadow-lg hover:bg-indigo-500 transition duration-200 disabled:bg-gray-600 disabled:cursor-not-allowed"
                        disabled={loading}
                    >
                        {loading ? (
                            <Loader2 className="w-5 h-5 mr-3 animate-spin" />
                        ) : (
                            <Search className="w-5 h-5 mr-3" />
                        )}
                        {loading ? 'PROCESSING...' : 'RUN QUERY'}
                    </button>
                </form>
                {error && (
                    <div className="mt-4 text-sm text-red-300 p-3 bg-red-900/40 rounded-lg border border-red-500 flex items-center font-mono shadow-inner">
                        <XCircle className="w-5 h-5 mr-2" />
                        ERROR LOG: {error}
                    </div>
                )}
            </div>

            {/* Results Section */}
            <div className="max-w-6xl mx-auto flex flex-col gap-8">
                
                {loading && <LoadingConsole />}
                
                {!loading && analysisResult && (
                    <>
                        {/* Statistics Row */}
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
                            <MetricDisplay 
                                icon={MapPin} 
                                title="TARGET REGION RECORDS" 
                                value={totalProperties.toLocaleString()} 
                                colorClass="text-indigo-400" 
                                borderColorClass="border-indigo-500/50"
                            />
                                <MetricDisplay 
                                icon={DollarSign} 
                                title="AVERAGE INDEX PRICE" 
                                value={getFormattedPrice(avgPrice)} 
                                colorClass="text-lime-400" 
                                borderColorClass="border-lime-500/50"
                            />
                            <MetricDisplay 
                                icon={Target} 
                                title="CURRENT DEMAND RATING" 
                                value={getFormattedDemand(latestDemand)} 
                                colorClass="text-yellow-400" 
                                borderColorClass="border-yellow-500/50"
                            />
                            <MetricDisplay 
                                icon={TrendingUp} 
                                title="TIME SERIES DEPTH (CYCLES)" 
                                value={chartData.length.toLocaleString()} 
                                colorClass="text-red-400" 
                                borderColorClass="border-red-500/50"
                            />
                        </div>

                        {/* Summary - The Mission Report */}
                        <div className="p-6 bg-gray-900 rounded-xl shadow-2xl border border-cyan-500/50">
                            <h2 className="text-xl font-bold text-cyan-400 mb-3 flex items-center">
                                <Rocket className="w-5 h-5 mr-2 text-cyan-400"/>
                                MISSION REPORT: ANALYSIS VERDICT
                            </h2>
                            <p className="text-gray-300 leading-relaxed font-sans">{summaryText}</p>
                        </div>

                        <ChartComponent />
                        <TableComponent />
                    </>
                )}
                
                {!loading && !analysisResult && !error && (
                    <div className="p-16 bg-gray-900 rounded-xl shadow-2xl border border-indigo-500/50 text-center">
                        <BarChart3 className="w-16 h-16 mx-auto mb-6 text-indigo-500" />
                        <p className="text-2xl font-bold text-gray-400">ATLAS CONSOLE INITIATED.</p>
                        <p className="text-lg font-mono text-gray-500 mt-2">Enter an asset location (e.g., Baner) in the command line above to generate your first intelligence report.</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default App;