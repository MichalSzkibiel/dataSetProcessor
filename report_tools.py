from config import language


def insert_table(doc, table_title):
    doc.add_heading(language[table_title], level=1)
    table = doc.add_table(rows=1, cols=2)
    header_cells = table.rows[0].cells
    header_cells[0].text = language['parameter_name']
    header_cells[1].text = language['parameter_value']
    return table


def insert_param_to_table(table, param_name, param_value):
    row_cells = table.add_row().cells
    row_cells[0].text = language[param_name]
    if isinstance(param_value, bool):
        row_cells[1].text = language[param_value]
    else:
        row_cells[1].text = str(param_value)


def percent_format(number):
    if isinstance(number, float):
        return '{:.2f}%'.format(number*100)
    else:
        return str(number)
