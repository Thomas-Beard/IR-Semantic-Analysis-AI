import sys
import requests
import subprocess
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QTextEdit, QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, 
                             QPushButton,QHBoxLayout, QTableWidget, QTableWidgetItem, QComboBox, 
                             QHeaderView, QStatusBar, QMessageBox, QTabWidget, QScrollArea, QStackedWidget)
from PyQt5.QtCore import QProcess, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sentence_transformers import SentenceTransformer
from xml.etree import ElementTree as ET
from pathlib import Path


SOLR_SELECT_URL = 'http://localhost:8990/solr/research-papers/select'
SOLR_QUERY_URL = 'http://localhost:8990/solr/research-papers/query'
BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def load_queries(qry_file="cran.qry.xml"):
    queries = {}
    root = ET.parse(qry_file).getroot()

    qrel_path = Path(__file__).resolve().parent / "cranqrel.trec.txt"
    rel_counts = {}
    with open(qrel_path, 'r') as f:
        for line in f:
            qid, _, _, rel = line.strip().split()
            if rel == '1':
                rel_counts[qid] = rel_counts.get(qid, 0) + 1

    for top in root.findall("top"):
        qid = top.findtext("num").strip()
        title = top.findtext("title").strip().replace('\n', ' ')
        count = rel_counts.get(qid, 0)
        marker = "⭐" if count >= 10 else ""
        queries[qid] = f"{marker} {title}"
    return queries


def load_qrels(qrel_path):
    qrels = {}
    with open(qrel_path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qid = qid.strip() 
            if rel == '1':
                qrels.setdefault(qid, set()).add(docid)
    return qrels


QREL_PATH = str(Path(__file__).resolve().parent / "cranqrel.trec.txt")
QRELS = load_qrels(QREL_PATH)


def evaluate_results(retrieved_ids, relevant_ids, k=10):
    def normalize(ids):
        return set(str(int(str(doc).strip())) for doc in ids if str(doc).strip().isdigit())

    retrieved_k = list(normalize(retrieved_ids))[:k]
    retrieved_set = normalize(retrieved_ids)
    relevant_set = normalize(relevant_ids)

    true_positives = len(set(retrieved_k) & relevant_set)
    precision = true_positives / k if k else 0
    recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0

    ap = 0
    hits = 0
    for i, doc_id in enumerate(retrieved_ids):
        doc_id_norm = str(int(str(doc_id).strip())) if str(doc_id).strip().isdigit() else None
        if doc_id_norm and doc_id_norm in relevant_set:
            hits += 1
            ap += hits / (i + 1)

    map_score = ap / len(relevant_set) if relevant_set else 0
    return precision, recall, map_score



class SearchThread(QThread):
    result_ready = pyqtSignal(list, str)
    error_capture = pyqtSignal(str)

    def __init__(self, query, mode):
        super().__init__()
        self.main_query = query
        self.paradigm_mode = mode

    def run(self):
        try:
            if self.paradigm_mode == "BM25 Paradigm":
                params = {
                    'q': f'title:{self.main_query} OR abstract:{self.main_query} OR text:{self.main_query} OR author:{self.main_query}',
                    'fl': 'id,title,score,abstract',
                    'rows': 50,
                    'wt': 'json'
                }
                self.query_params = params 
                response = requests.get(SOLR_SELECT_URL, params=params)
            elif self.paradigm_mode == "Hybrid Paradigm (BM25 + Vector)":
                vector = BERT_MODEL.encode(self.main_query).tolist()
                vec_str = ','.join([str(round(x, 6)) for x in vector])
                params = {
                    'q': f'title:{self.main_query} OR abstract:{self.main_query} OR text:{self.main_query} OR author:{self.main_query}',
                    'fl': 'id,title,score,abstract',
                    'rows': 50,
                    'wt': 'json',
                    'rq': '{!rerank reRankQuery=$rvec reRankDocs=100 reRankWeight=100.0}',
                    'rvec': f'{{!knn f=vector topK=100}}[{vec_str}]'
                }
                self.query_params = params 
                response = requests.get(SOLR_SELECT_URL, params=params)
            elif self.paradigm_mode == "Semantic Paradigm (Vectors)":
                vector = BERT_MODEL.encode(self.main_query)

                #print(f"[DEBUG] Vector length: {len(vector)}")
                if len(vector) != 384:
                    self.error_capture.emit(f"Invalid vector length: {len(vector)} (expected 384)")
                    return

                vec_str = ','.join([str(round(x, 6)) for x in vector])
                params = {
                    'q': f'{{!knn f=vector topK=50}}[{vec_str}]',
                    'fl': 'id,title,score,abstract',
                    'rows': 50,
                    'wt': 'json'
                }
                self.query_params = params 
                response = requests.get(SOLR_SELECT_URL, params=params)
            else:
                #response = requests.get(SOLR_SELECT_URL, params=params)
                self.error_capture.emit("Invalid mode.")
                return

            response.raise_for_status()
            docs = response.json()['response']['docs']
            self.result_ready.emit(docs, self.paradigm_mode)
        except Exception as e:
            self.error_capture.emit(str(e))

class GraphsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.switch_to_scores = QPushButton("Show Score Distribution")
        self.switch_to_metrics = QPushButton("Show Evaluation Metrics")
        self.switch_to_scores.clicked.connect(self.show_scores)
        self.switch_to_metrics.clicked.connect(self.show_metrics)

        layout.addWidget(self.switch_to_scores)
        layout.addWidget(self.switch_to_metrics)

        self.page_stack = QStackedWidget()
        layout.addWidget(self.page_stack)

        self.score_page = QWidget()
        self.score_layout = QVBoxLayout(self.score_page)
        self.score_figure = Figure(figsize=(10, 6))
        self.score_canvas = FigureCanvas(self.score_figure)
        self.score_layout.addWidget(self.score_canvas)
        self.page_stack.addWidget(self.score_page)

        self.metric_page = QWidget()
        self.metric_layout = QVBoxLayout(self.metric_page)
        self.metric_figure, self.metric_ax = plt.subplots(figsize=(10, 6))
        self.metric_canvas = FigureCanvas(self.metric_figure)
        self.metric_layout.addWidget(self.metric_canvas)
        self.page_stack.addWidget(self.metric_page)

        self.setLayout(layout)
        self.page_stack.setCurrentIndex(0)

    def show_scores(self):
        self.page_stack.setCurrentIndex(0)

    def show_metrics(self):
        self.page_stack.setCurrentIndex(1)

    def plot_score_distribution(self, scores):
        self.score_figure.clear()
        ax = self.score_figure.add_subplot(111)
        ax.hist(scores, bins=10, color='skyblue', edgecolor='black')
        ax.set_title('Search Score Distribution')
        ax.set_xlabel('Relevance Score')
        ax.set_ylabel('Number of Documents')
        self.score_canvas.draw()

    def plot_metric_comparison(self, metrics):
        self.metric_ax.clear()

        paradigms = list(metrics.keys())
        precision = [metrics[p].get('P@10', 0) for p in paradigms]
        recall = [metrics[p].get('Recall', 0) for p in paradigms]
        map_scores = [metrics[p].get('MAP', 0) for p in paradigms]

        if not any(precision + recall + map_scores):
            self.metric_ax.set_title("No non-zero metrics to display.")
            self.metric_canvas.draw()
            return

        bar_width = 0.25
        x = np.arange(len(paradigms))

        bars1 = self.metric_ax.bar(x, precision, width=bar_width, label='Precision@50')
        bars2 = self.metric_ax.bar(x + bar_width, recall, width=bar_width, label='Recall')
        bars3 = self.metric_ax.bar(x + 2 * bar_width, map_scores, width=bar_width, label='MAP')

        self.metric_ax.set_title('Evaluation Metrics per Paradigm')
        self.metric_ax.set_xticks(x + bar_width)
        self.metric_ax.set_xticklabels(paradigms, rotation=15)
        self.metric_ax.set_ylim(0, 1)
        self.metric_ax.legend()

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                self.metric_ax.annotate(f"{height:.2f}",
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom')

        self.metric_canvas.draw()

class SearchTab(QWidget):
    def __init__(self, status_bar, graphs_tab):
        super().__init__()
        self.status_bar = status_bar
        self.graphs_tab = graphs_tab 
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        button_panel = QVBoxLayout()
        info_panel = QVBoxLayout()

        # self.label = QLabel('Enter your search query:')
        # layout.addWidget(self.label)

        # self.query_input = QLineEdit()
        # layout.addWidget(self.query_input)

        self.label = QLabel('Select Query:')
        layout.addWidget(self.label)
        xml_path = str(Path(__file__).resolve().parent / "cran.qry.xml")
        self.query_dict = load_queries(xml_path)
        self.query_input = QComboBox()
        for qid, qtext in self.query_dict.items():
            display_text = f"{qid}: {qtext.strip()}"
            self.query_input.addItem(display_text, userData=(qid, qtext))
        layout.addWidget(self.query_input)


        self.search_mode = QComboBox()
        self.search_mode.addItems(["BM25 Paradigm", "Semantic Paradigm (Vectors)", "Hybrid Paradigm (BM25 + Vector)"])
        layout.addWidget(self.search_mode)

        self.search_button = QPushButton('Search')
        self.search_button.clicked.connect(self.run_search)
        layout.addWidget(self.search_button)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(['ID', 'Title', 'Score'])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setColumnWidth(0, 80)
        self.results_table.setColumnWidth(2, 80)

        self.results_table.cellDoubleClicked.connect(self.show_abstract)

        # layout.addWidget(self.results_table, stretch=1)
        layout.addWidget(self.results_table)
        layout.setStretchFactor(self.results_table, 2)



        layout.addLayout(button_panel)
        layout.addLayout(info_panel)
        self.setLayout(layout)
        self.doc_abstracts = {}

    def evaluate_all_paradigms(self, query_id, query_text):
        relevant_docs = QRELS.get(str(query_id).strip(), set())
        metrics = {}

        for mode in ["BM25 Paradigm", "Semantic Paradigm (Vectors)", "Hybrid Paradigm (BM25 + Vector)"]:
            if mode == "BM25 Paradigm":
                params = {
                    'q': f'title:{query_text} OR abstract:{query_text} OR text:{query_text} OR author:{query_text}',
                    'fl': 'id,title,score,abstract',
                    'rows': 50,
                    'wt': 'json'
                }
            elif mode == "Semantic Paradigm (Vectors)":
                vector = BERT_MODEL.encode(query_text).tolist()
                vec_str = ','.join([str(round(x, 6)) for x in vector])
                params = {
                    'q': f'{{!knn f=vector topK=50}}[{vec_str}]',
                    'fl': 'id,title,score,abstract',
                    'rows': 50,
                    'wt': 'json'
                }
            elif mode == "Hybrid Paradigm (BM25 + Vector)":
                vector = BERT_MODEL.encode(query_text).tolist()
                vec_str = ','.join([str(round(x, 6)) for x in vector])
                params = {
                    'q': f'title:{query_text} OR abstract:{query_text} OR text:{query_text} OR author:{query_text}',
                    'fl': 'id,title,score,abstract',
                    'rows': 50,
                    'wt': 'json',
                    'rq': '{!rerank reRankQuery=$rvec reRankDocs=100 reRankWeight=100.0}',
                    'rvec': f'{{!knn f=vector topK=100}}[{vec_str}]'
                }

            try:
                response = requests.get(SOLR_SELECT_URL, params=params)
                response.raise_for_status()
                docs = response.json()['response']['docs']
                doc_ids = [doc.get('id', '') for doc in docs]
                p, r, m = evaluate_results(doc_ids, relevant_docs, k=50)
                metrics[mode] = {'P@10': p, 'Recall': r, 'MAP': m}
            except Exception as e:
                metrics[mode] = {'P@10': 0, 'Recall': 0, 'MAP': 0}
                print(f"[DEBUG] Error evaluating {mode}: {e}")

        self.graphs_tab.plot_metric_comparison(metrics)
        print("[DEBUG] Available qrel keys (sample):", sorted(QRELS.keys())[:10])
        print("[DEBUG] Total queries with qrels:", len(QRELS))

        print("[DEBUG] Using query_id:", query_id)



    def run_search(self):
        if self.query_input.currentIndex() == -1:
            self.status_bar.showMessage('Please select a query.')
            return

        query_id, raw_query_text = self.query_input.currentData()
        query_text = raw_query_text.replace("⭐", "").strip()

        self.current_query_id = query_id
        mode = self.search_mode.currentText()

        self.search_button.setEnabled(False)
        self.status_bar.showMessage(f'Searching for Query #{query_id}...')

        self.search_thread = SearchThread(query_text, mode)
        self.search_thread.result_ready.connect(self.display_results)
        self.search_thread.error_capture.connect(self.handle_error)
        self.search_thread.start()
        self.evaluate_all_paradigms(query_id, query_text)




    def display_results(self, docs, mode_label):
        self.results_table.setRowCount(len(docs))
        self.results_table.resizeRowsToContents()
        # self.results_table.resizeColumnsToContents()

        self.doc_abstracts = {}

        for doc in docs[:5]:
            print("[DEBUG] Returned ID:", doc.get("id"))

        for row, doc in enumerate(docs):
            doc_id = doc.get('id', 'N/A')
            if isinstance(doc_id, list):
                doc_id = ' '.join(doc_id)

            title = doc.get('title', 'No Title')
            if isinstance(title, list):
                title = ' '.join(title)

            relevance_score = doc.get('score', 0)
            score_str = f"{relevance_score:.3f}" if isinstance(relevance_score, (int, float)) else str(relevance_score)

            id_item = QTableWidgetItem(doc_id)
            title_item = QTableWidgetItem(title)
            score_item = QTableWidgetItem(score_str)

            self.results_table.setItem(row, 0, id_item)
            self.results_table.setItem(row, 1, title_item)
            self.results_table.setItem(row, 2, score_item)

            abstract = doc.get('abstract', 'No abstract available.')
            if isinstance(abstract, list):
                abstract = ' '.join(abstract)
            self.doc_abstracts[row] = abstract


        self.status_bar.showMessage(f"Found {len(docs)} documents ({mode_label}).")
        self.search_button.setEnabled(True)

        scores = [doc.get('score', 0) for doc in docs]
        self.graphs_tab.plot_score_distribution(scores)
        
    def handle_error(self, msg):
        self.status_bar.showMessage(f"Error: {msg}")
        self.search_button.setEnabled(True)

    def show_abstract(self, row, column):
        abstract = self.doc_abstracts.get(row, 'No abstract available.')
        if isinstance(abstract, list):
            abstract = ' '.join(abstract)
        QMessageBox.information(self, "Document Abstract", abstract)



class SolrProcessWidget(QWidget):
    def __init__(self, bat_file_path="temp.bat"):
        super().__init__()
        self.bat_file_path = bat_file_path
        self.process = QProcess(self)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        button_panel = QVBoxLayout()
        info_panel = QVBoxLayout()

        self.run_button = QPushButton("Connect to Solr")
        self.run_button.clicked.connect(self.run_script)
        button_panel.addWidget(self.run_button)

        self.disconnect_button = QPushButton("Disconnect from Solr")
        self.disconnect_button.clicked.connect(self.stop_solr)
        button_panel.addWidget(self.disconnect_button)

        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        info_panel.addWidget(self.output_console)

        self.user_label = QLabel(f"Current User: {os.getlogin()}")
        self.datetime_label = QLabel(f"Time/Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.status_label = QLabel("Connection Status: Unknown")
        info_panel.addWidget(self.user_label)
        info_panel.addWidget(self.datetime_label)
        info_panel.addWidget(self.status_label)

        layout.addLayout(button_panel)
        layout.addLayout(info_panel)

        self.setLayout(layout)

        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)


    def run_script(self):
        self.output_console.clear()

        self.output_console.append("Killing existing Java processes...\n")
        try:
            subprocess.run(["taskkill", "/F", "/IM", "java.exe"], check=True, capture_output=True)
            self.output_console.append("Terminated java.exe processes.\n")
        except subprocess.CalledProcessError as e:
            self.output_console.append(f"Could not kill java.exe processes or none were running.\n")

        if not Path(self.bat_file_path).exists():
            self.output_console.append(f"Error: Cannot find {self.bat_file_path}")
            self.status_label.setText("Connection Status: Disconnected")
            return

        self.output_console.append(f"Running: {self.bat_file_path}\n")
        self.status_label.setText("Connection Status: Starting...")
        self.process.start("cmd.exe", ["/c", self.bat_file_path])


    def stop_solr(self):
        self.output_console.append("\nStopping Solr...\n")
        stop_script = str(Path(self.bat_file_path).parent / "solr-9.5.0" / "bin" / "solr.cmd")
        if not Path(stop_script).exists():
            self.output_console.append("Error: Could not find solr.cmd to stop Solr.\n")
            self.status_label.setText("Connection Status: Unknown")
            return

        self.status_label.setText("Connection Status: Disconnecting...")
        self.process.start("cmd.exe", ["/c", stop_script, "stop", "-all"])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        self.output_console.append(str(data, encoding='utf-8'))

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        self.output_console.append(str(data, encoding='utf-8'))

    def process_finished(self):
        self.output_console.append("\nProcess finished.")
        try:
            response = requests.get("http://localhost:8990/solr/admin/info/system", timeout=5)
            if response.status_code == 200:
                self.status_label.setText("Connection Status: ZooKeeper & Solr Online")
                self.output_console.append("Solr is online. Proceeding to create collection...\n")
                self.run_create_collection()
            else:
                self.status_label.setText("Connection Status: Disconnected")
        except Exception:
            self.status_label.setText("Connection Status: Disconnected")

    def run_create_collection(self):
        try:
            from subprocess import run
            script_path = Path(__file__).resolve().parent / "collection_updates.py"

            python_exe = sys.executable

            result = run(
                [python_exe, str(script_path)],
                capture_output=True,
                text=True
            )

            self.output_console.append("Create Collection Output:\n" + result.stdout)
            if result.stderr:
                self.output_console.append("Errors:\n" + result.stderr)
        except Exception as e:
            self.output_console.append(f"Failed to run create_collection.py: {e}")


class InfoTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h2>Welcome to the Semantic IR System</h2>

        <h3>Purpose:</h3>
        <p>
        This software allows users to explore and compare traditional and modern information retrieval paradigms.
        It supports fixed query search using BM25, semantic search using vector embeddings, and a hybrid approach combining both.
        The goal is to visualize search results, analyze scoring behavior, examine how relevance shifts across query types and search paradigms.
        Furthermore, users such as researchers and students (both a common beneficiary searching for research papers) can use this product to find the most relatable paper for their research. 
        
        A surprising result was found during the development of this product, where the semantic paradigm (utilising a BERT model trained and tested for IR systems) often produced more accurate results compared to the hybrid and BM25 paradigms respectfully. This can be viewed in the Graphs tab when comparing evaluation metrics.
        Queries marked with a "⭐" have the most relevant documents according to relevance judgement document "cranqrel.trex.txt". 
        </p>
        
        <p>TREC Collection, Relevance Judgement and Fixed Queries utilised from: https://github.com/oussbenk/cranfield-trec-dataset/tree/main</p>
        
        <h3>Limitations:</h3>
        <ul>
            <li>The cranfield TREC collection only contains 1,400 documents, which results in the data provided for analysis to not be 100% accurate, but can still be reliable for analysis & finding a specific research papers for the users.</li>
            <li>Queries beginning with the word "what" for the BM25 paradigm tend to output incorrect results. Although this isn't common, it may be an output for longer queries especially when they are generic or vague qeuries. I.e. "What is..", which match a wide range of documents in BM25 due to keyword overlap. Furthermore, "what" queries are often weak in semantics for the BM25 paradigm. A solution for this could be stripping the queries with phrases that contain "what is" or "what are".  </li>
            <li>Recent releases of SOLR do not work on the developers machine, this could be an internal issue. This limits the developer to features in solr 9.5.0</li>
        </ul>

        <h3>Instructions:</h3>
        <ul>
            <li>Go to the <b>Solr Setup</b> tab and click "Connect to Solr".</li>
            <li>Use the <b>Search</b> tab to enter queries and view ranked results.</li>
            <li>Explore the <b>Graphs</b> tab to visualize relevance score distributions.</li>
            <li>Double-click a document in the results table to read its abstract.</li>
        </ul>

        <h3>Dependencies:</h3>
        <ul>
            <li><b>ZooKeeper</b>: Standalone mode used for SolrCloud coordination.</li>
            <li><b>Java JDK</b>: Java 17 is required. Must include <code>java.exe</code> in the <code>bin</code> directory.</li>
            <li><b>Apache Solr</b>: Version 9.5.0, running in cloud mode with reranking support.</li>
        </ul>

        <p>
        Ensure all directories (e.g., <code>jdk-17</code>, <code>zookeeper</code>, <code>solr-9.5.0</code>) and the <code>temp.bat</code> file are present in the main application folder.
        </p>
        """)

        layout.addWidget(info_text)
        self.setLayout(layout)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Semantic IR System')
        self.resize(900, 600)
        self.setMinimumSize(900, 600)

        # layout = QHBoxLayout()
        self.tabs = QTabWidget()

        self.status_bar = QStatusBar()
        self.graphs_tab = GraphsTab()
        self.search_tab = SearchTab(self.status_bar, self.graphs_tab)

        self.tabs.addTab(InfoTab(), "Information")
        root_bat_path = str(Path(__file__).resolve().parents[1] / "temp.bat")
        self.tabs.addTab(SolrProcessWidget(bat_file_path=root_bat_path), "Solr Setup")
        self.tabs.addTab(self.search_tab, "Search")
        self.tabs.addTab(self.graphs_tab, "Graphs")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(self.status_bar)
        self.setLayout(main_layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())