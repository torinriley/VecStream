import pytest
from vector_db.command_interface import CommandInterface

def test_vector_add_command():
    cmd_interface = CommandInterface()
    
    # Test basic vector add
    result = cmd_interface.execute_command("VADD test1 1.0 2.0 3.0")
    assert result == "OK"
    
    # Test vector add with metadata
    result = cmd_interface.execute_command('VADD test2 4.0 5.0 6.0 METADATA {"category": "test"}')
    assert result == "OK"
    
    # Test invalid command
    result = cmd_interface.execute_command("VADD")
    assert result.startswith("ERROR")

def test_vector_get_command():
    cmd_interface = CommandInterface()
    cmd_interface.execute_command("VADD test3 1.0 2.0 3.0")
    
    # Test getting existing vector
    result = cmd_interface.execute_command("VGET test3")
    assert "[1.0, 2.0, 3.0]" in result
    
    # Test getting non-existent vector
    result = cmd_interface.execute_command("VGET nonexistent")
    assert result == "(nil)"

def test_vector_delete_command():
    cmd_interface = CommandInterface()
    cmd_interface.execute_command("VADD test4 1.0 2.0 3.0")
    
    # Test deleting existing vector
    result = cmd_interface.execute_command("VDEL test4")
    assert result == "1"
    
    # Test getting deleted vector (should be gone)
    result = cmd_interface.execute_command("VGET test4")
    assert result == "(nil)"

def test_vector_search_command():
    cmd_interface = CommandInterface()
    cmd_interface.execute_command("VADD vec1 1.0 0.0 0.0")
    cmd_interface.execute_command("VADD vec2 0.0 1.0 0.0")
    cmd_interface.execute_command("VADD vec3 0.5 0.5 0.0")
    
    # Test basic search
    result = cmd_interface.execute_command("VSEARCH 1.0 0.0 0.0 LIMIT 2")
    assert "vec1" in result
    
    # Test search with different metric
    result = cmd_interface.execute_command("VSEARCH 1.0 0.0 0.0 LIMIT 2 METRIC dot")
    assert "vec1" in result

def test_vector_count_command():
    cmd_interface = CommandInterface()
    cmd_interface.execute_command("VADD item1 1.0 1.0 1.0")
    cmd_interface.execute_command("VADD item2 2.0 2.0 2.0")
    
    result = cmd_interface.execute_command("VCOUNT")
    assert result == "2"

def test_vector_list_command():
    cmd_interface = CommandInterface()
    cmd_interface.execute_command("VADD list1 1.0 1.0 1.0")
    cmd_interface.execute_command("VADD list2 2.0 2.0 2.0")
    
    result = cmd_interface.execute_command("VLIST")
    assert "list1" in result
    assert "list2" in result
