import sys
import requests
import subprocess
import os
import datetime
from PyQt5.QtWidgets import (QTextEdit, QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, 
                             QPushButton,QHBoxLayout, QTableWidget, QTableWidgetItem, QComboBox, 
                             QHeaderView, QStatusBar, QMessageBox, QTabWidget)
from PyQt5.QtCore import QProcess
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sentence_transformers import SentenceTransformer
from xml.etree import ElementTree as ET
from pathlib import Path

SOLR_SELECT_URL = 'http://localhost:8990/solr/research-papers/select'
SOLR_QUERY_URL = 'http://localhost:8990/solr/research-papers/query'
BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')


# Loading the queries from the TREC docs when app is initialised
def load_queries(qry_file="cran.qry.xml"):
    queries = {}
    root = ET.parse(qry_file).getroot()
    for top in root.findall("top"):
        qid = top.findtext("num").strip()
        title = top.findtext("title").strip().replace('\n', ' ')
        queries[qid] = title
    return queries

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
        layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_score_distribution(self, scores):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.hist(scores, bins=10, color='skyblue', edgecolor='black')
        ax.set_title('Search Score Distribution')
        ax.set_xlabel('Relevance Score')
        ax.set_ylabel('Number of Documents')
        self.canvas.draw()


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

    def run_search(self):
        if self.query_input.currentIndex() == -1:
            self.status_bar.showMessage('Please select a query.')
            return

        query_id, query_text = self.query_input.currentData()
        self.current_query_id = query_id  # Save for evaluation
        mode = self.search_mode.currentText()

        self.search_button.setEnabled(False)
        self.status_bar.showMessage(f'Searching for Query #{query_id}...')

        self.search_thread = SearchThread(query_text, mode)
        self.search_thread.result_ready.connect(self.display_results)
        self.search_thread.error_capture.connect(self.handle_error)
        self.search_thread.start()

    def display_results(self, docs, mode_label):
        self.results_table.setRowCount(len(docs))
        self.results_table.resizeRowsToContents()
        # self.results_table.resizeColumnsToContents()

        self.doc_abstracts = {}

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

        <h3>Purpose</h3>
        <p>
        This software allows users to explore and compare traditional and modern information retrieval paradigms.
        It supports fixed query search using BM25, semantic search using vector embeddings, and a hybrid approach combining both.
        The goal is to visualize search results, analyze scoring behavior, examine how relevance shifts across query types and search paradigms.
        Furthermore, users such as researchers and students (both a common beneficiary searching for research papers) can use this product to find the most relatable paper for their research. 
        However, due to the main limitation where the collection only contains 1,400 documents of topics available; Aerodynamics, Thermal analysis, Structual mechanics, Hypersonic flight & Gas dynamics.
        This means the product will only return a max amount of search results (i.e. 50 results max) & only the papers related to the topics in the collection which means the data provided may not be 100% accurate, but can still be reliable for analysis & finding a specific research papers for the users.
        </p>

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