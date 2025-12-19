import json
from pathlib import Path

import pytest

from code_transformer.adapters.jsonl_reader import JSONLReader
from code_transformer.domain.exceptions import InvalidInputError
from code_transformer.domain.models import CodeSnippet


@pytest.fixture
def valid_json_file(tmp_path: Path) -> Path:
    """Create a valid json file for testing."""
    file_path = tmp_path / "valid.jsonl"

    data = [
        {
            "id": "test_001",
            "file_path": "test.go",
            "language": "go",
            "content": "func main() \n return 'hi'"
        },
        {
            "id": "test_002",
            "file_path": "test.py",
            "language": "python",
            "content": "def main(): \n return 'hello world'"
        },
        {
            "id": "test_003",
            "file_path": "test.js",
            "language": "javascript",
            "content": "function main() { return 'hi'; }"
        }
    ]

    with open(file_path, mode="w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return file_path

@pytest.fixture
def json_with_optional_fields(tmp_path: Path) -> Path:
    """Create a json file with optional fields for testing."""
    file_path = tmp_path / "optional.jsonl"

    data = [
              {
                  "id": "opt_001",
                  "file_path": "handler.go",
                  "language": "go",
                  "content": "func Handle() {}",
                  "version": "v1.2.3",
                  "git_hash": "abc123",
                  "chunk_name": "Handle",
                  "chunk_type": "function",
                  "metadata": {
                      "start_line": 10,
                      "end_line": 12,
                      "receiver": None,
                      "type_kind": "function"
                  },
                  "created_at": "2025-01-15T10:30:00Z"
              }
            ]

    with open(file_path, mode="w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return file_path

@pytest.fixture
def invalid_json_file(tmp_path: Path) -> Path:
    """Create an invalid json file for testing."""
    file_path = tmp_path / "invalid.jsonl"

    invalid_content = "invalid content"
    with open(file_path, mode="w") as f:
        f.write(invalid_content + "\n")

    return file_path

@pytest.fixture
def empty_json_file(tmp_path: Path) -> Path:
    """Create an empty json file for testing."""
    file_path = tmp_path / "empty.jsonl"
    file_path.touch()
    return file_path

@pytest.fixture
def json_with_missing_fields(tmp_path: Path) -> Path:
    """Create a json file with missing required fields for testing."""
    file_path = tmp_path / "missing_fields.jsonl"

    data = [
        {
            "id": "miss_001",
            "file_path": "missing.go",
            "language": "go"
        }
    ]

    with open(file_path, mode="w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return file_path

@pytest.fixture
def json_with_empty_lines(tmp_path: Path) -> Path:
    """Create a json file with empty lines for testing."""
    file_path = tmp_path / "empty_lines.jsonl"

    data = [
        {
            "id": "empty_001",
            "file_path": "empty.go",
            "language": "go",
            "content": "func Empty() {}"
        },
        {},
        {
            "id": "empty_002",
            "file_path": "empty2.py",
            "language": "python",
            "content": "def empty(): pass"
        }
    ]

    with open(file_path, mode="w") as f:
        for item in data:
            if item:
                f.write(json.dumps(item) + "\n")
            else:
                f.write("\n")

    return file_path

class TestJSONLReaderInit:
    """Test JSONLReader initialization."""

    def test_with_string_path(self, valid_json_file: Path) -> None:
        """Test JSONLReader with string file path."""
        reader = JSONLReader(str(valid_json_file))
        assert reader.file_path == valid_json_file
        assert isinstance(reader.file_path, Path)

    def test_init_with_path_object(self, valid_json_file: Path) -> None:
        """Test initialization with Path object."""
        reader = JSONLReader(valid_json_file)
        assert reader.file_path == valid_json_file

    def test_init_with_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            JSONLReader("")

    def test_init_with_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            JSONLReader(None)


class TestJSONLReaderRead:
    """Test JSONLReader.read() async generator method."""

    @pytest.mark.asyncio
    async def test_read_all_snippets_from_valid_file(self, valid_json_file: Path) -> None:
        """Test reading all snippets from valid JSONL file."""
        reader = JSONLReader(valid_json_file)

        # Collect all snippets using async for
        snippets = [snippet async for snippet in reader.read()]

        # Verify count
        assert len(snippets) == 3

        # Verify first snippet
        assert snippets[0].id == "test_001"
        assert snippets[0].file_path == "test.go"
        assert snippets[0].language == "go"
        assert "main" in snippets[0].content

        # Verify all are CodeSnippet instances
        assert all(isinstance(s, CodeSnippet) for s in snippets)

    @pytest.mark.asyncio
    async def test_read_snippets_with_optional_fields(
        self, json_with_optional_fields: Path
    ) -> None:
        """Test reading snippets with optional fields preserved."""
        reader = JSONLReader(json_with_optional_fields)

        snippets = [snippet async for snippet in reader.read()]

        assert len(snippets) == 1
        snippet = snippets[0]

        # Check optional fields are preserved
        assert snippet.version == "v1.2.3"
        assert snippet.git_hash == "abc123"
        assert snippet.chunk_name == "Handle"
        assert snippet.chunk_type == "function"
        assert snippet.metadata["start_line"] == 10
        assert snippet.created_at == "2025-01-15T10:30:00Z"

    @pytest.mark.asyncio
    async def test_read_invalid_json_raises_error(self, invalid_json_file: Path) -> None:
        """Test that invalid JSON raises InvalidInputError."""
        reader = JSONLReader(invalid_json_file)

        with pytest.raises(InvalidInputError, match="not a valid JSONL"):
            # Must consume the generator to trigger the error
            [snippet async for snippet in reader.read()]

    @pytest.mark.asyncio
    async def test_read_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that missing file raises InvalidInputError."""
        nonexistent = tmp_path / "does_not_exist.jsonl"
        reader = JSONLReader(nonexistent)

        with pytest.raises(InvalidInputError, match="does not exist"):
            [snippet async for snippet in reader.read()]

    @pytest.mark.asyncio
    async def test_read_empty_file_returns_empty_list(self, empty_json_file: Path) -> None:
        """Test that empty file yields no snippets."""
        reader = JSONLReader(empty_json_file)

        snippets = [snippet async for snippet in reader.read()]

        assert snippets == []
        assert len(snippets) == 0


class TestJSONLReaderValidate:
    """Test JSONLReader.validate method."""

    @pytest.mark.asyncio
    async def test_validate_returns_true_for_valid_file(self, valid_json_file: Path) -> None:
        """Test that valid file passes validation."""
        reader = JSONLReader(valid_json_file)
        is_valid = await reader.validate()
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_returns_false_for_invalid_json(self, invalid_json_file: Path) -> None:
        """Test that invalid JSON fails validation."""
        reader = JSONLReader(invalid_json_file)
        is_valid = await reader.validate()
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_returns_false_for_missing_file(self, tmp_path: Path) -> None:
        """Test that missing file fails validation."""
        nonexistent = tmp_path / "missing.jsonl"
        reader = JSONLReader(nonexistent)
        is_valid = await reader.validate()
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_returns_false_for_missing_required_fields(
        self, json_with_missing_fields: Path
    ) -> None:
        """Test that snippets with missing required fields fail validation."""
        reader = JSONLReader(json_with_missing_fields)
        is_valid = await reader.validate()
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_returns_true_for_optional_fields(
        self, json_with_optional_fields: Path
    ) -> None:
        """Test that optional fields don't affect validation."""
        reader = JSONLReader(json_with_optional_fields)
        is_valid = await reader.validate()
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_empty_file(self, empty_json_file: Path) -> None:
        """Test validation of empty file."""
        reader = JSONLReader(empty_json_file)
        is_valid = await reader.validate()
        # Current implementation: empty file passes validation
        # because loop breaks immediately (no lines to validate)
        assert is_valid is True
