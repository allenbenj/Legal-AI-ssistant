"""Simplified PyQt6 interface for the Legal AI System.




@dataclass
class Document:


    id: str
    filename: str
    status: str = "pending"
    progress: float = 0.0

    file_size: int = 0
    doc_type: str = "Unknown"


            return None
        value = self.documents.iat[index.row(), index.column()]
        return str(value)


    sys.exit(app.exec())


if __name__ == "__main__":
    main()
