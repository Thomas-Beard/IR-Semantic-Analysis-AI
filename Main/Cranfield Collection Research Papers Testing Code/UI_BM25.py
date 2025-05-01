from pathlib import Path

import sys
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QStatusBar,
    QMessageBox, QTabWidget
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sentence_transformers import SentenceTransformer

SOLR_URL = 'http://localhost:8983/solr/research-papers/select'
BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

class SearchThread(QThread):
    result_ready = pyqtSignal(list, str)
    error = pyqtSignal(str)

    def __init__(self, query, mode):
        super().__init__()
        self.query = query
        self.mode = mode

    def run(self):
        try:
            if self.mode == "BM25 Only":
                params = {
                    'q': f'_text_:({self.query})',
                    'fl': 'id,title,score,abstract',
                    'rows': 20,
                    'wt': 'json'
                }
            elif self.mode == "Hybrid (BM25 + Vector)":
                vector = BERT_MODEL.encode(self.query).tolist()
                vec_str = ','.join([str(round(x, 6)) for x in vector])
                params = {
                    'q': f'_text_:({self.query})',
                    'fl': 'id,title,score,abstract',
                    'rows': 100,
                    'wt': 'json',
                    'rq': '{!rerank reRankQuery=$rvec reRankDocs=100 reRankWeight=1.0}',
                    'rvec': f'{{!knn f=vector topK=5}}[{vec_str}]'
                }
            elif self.mode == "Vector-Only (Semantic Search)":
                vector = BERT_MODEL.encode(self.query).tolist()
                vec_str = ','.join([str(round(x, 6)) for x in vector])
                params = {
                    'q': f'{{!knn f=vector topK=10}}[{vec_str}]',
                    'fl': 'id,title,score,abstract',
                    'wt': 'json'
                }
            else:
                self.error.emit("Invalid mode.")
                return

            response = requests.get(SOLR_URL, params=params)
            response.raise_for_status()
            docs = response.json()['response']['docs']
            self.result_ready.emit(docs, self.mode)
        except Exception as e:
            self.error.emit(str(e))

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

        self.label = QLabel('Enter your search query:')
        layout.addWidget(self.label)

        self.query_input = QLineEdit()
        layout.addWidget(self.query_input)

        self.search_mode = QComboBox()
        self.search_mode.addItems(["BM25 Only", "Hybrid (BM25 + Vector)", "Vector-Only (Semantic Search)"])
        layout.addWidget(self.search_mode)

        self.search_button = QPushButton('Search')
        self.search_button.clicked.connect(self.run_search)
        layout.addWidget(self.search_button)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(['ID', 'Title', 'Score'])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.cellDoubleClicked.connect(self.show_abstract)
        layout.addWidget(self.results_table)

        self.setLayout(layout)
        self.doc_abstracts = {}

    def run_search(self):
        query = self.query_input.text().strip()
        mode = self.search_mode.currentText()

        if not query:
            self.status_bar.showMessage('Please enter a search query.')
            return

        self.search_button.setEnabled(False)
        self.status_bar.showMessage('Searching...')

        self.search_thread = SearchThread(query, mode)
        self.search_thread.result_ready.connect(self.display_results)
        self.search_thread.error.connect(self.handle_error)
        self.search_thread.start()

    def display_results(self, docs, mode_label):
        self.results_table.setRowCount(len(docs))
        self.doc_abstracts = {}

        for row, doc in enumerate(docs):
            id_item = QTableWidgetItem(doc.get('id', 'N/A'))
            title_item = QTableWidgetItem(doc.get('title', 'No Title'))
            score_item = QTableWidgetItem(f"{doc.get('score', 0):.3f}")

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

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Semantic IR System')
        self.resize(1100, 700)

        layout = QVBoxLayout()

        self.tabs = QTabWidget()

        self.status_bar = QStatusBar()
        self.graphs_tab = GraphsTab()
        self.search_tab = SearchTab(self.status_bar, self.graphs_tab)

        self.tabs.addTab(self.search_tab, "Search")
        self.tabs.addTab(self.graphs_tab, "Graphs")

        layout.addWidget(self.tabs)
        layout.addWidget(self.status_bar)

        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())