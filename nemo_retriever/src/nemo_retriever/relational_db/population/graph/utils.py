import regex
from datetime import timezone

import pandas as pd


def flat_list_recursive(nested_list):
    output = []
    for i in nested_list:
        if isinstance(i, list):
            temp = flat_list_recursive(i)
            for j in temp:
                output.append(j)
        else:
            output.append(i)
    return output


def remove_redundant_parentheses(text):
    r = "s/(\(|^)\K(\((((?2)|[^()])*)\))(?=\)|$)/\\3/"
    if r[0] != "s":
        raise SyntaxError('Missing "s"')
    d = r[1]
    r = r.split(d)
    if len(r) != 4:
        raise SyntaxError("Wrong number of delimiters")
    flags = 0
    count = 1
    for f in r[3]:
        if f == "g":
            count = 0
        else:
            flags |= {
                "i": regex.IGNORECASE,
                "m": regex.MULTILINE,
                "s": regex.DOTALL,
                "x": regex.VERBOSE,
            }[f]
    s = r[2]
    r = r[1]
    # z = 0

    while 1:
        m = regex.subn(r, s, text, count, flags)
        text = m[0]
        if m[1] == 0:
            break

    return text


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]





def load_tables(filepath_or_buffer, is_csv=True):
    types = {
        "database": "category",
        "schema": "category",
        "table_name": "string",
        "table_type": "category",
        "row_count": "Int64",
        "size": "Int64",
        "retention_time": "Int8",
        "created": "string",
        "last_altered": "string",
        "comment": "string",
    }
    if is_csv:
        df = pd.read_csv(
            filepath_or_buffer,
            dtype=types,
        )
        # read_csv can get columns list, but then we need to ignore the header to "fill" missing columns
        # so this approch feels much more safe
        for key in types.keys():
            if key not in df:
                df[key] = pd.NA
    else:
        if not isinstance(filepath_or_buffer, pd.DataFrame):
            df = pd.DataFrame(filepath_or_buffer)
        else:
            df = filepath_or_buffer

        # Happens for temp schema after reset
        if df.empty:
            return df

        if "row_count" not in df.columns:
            df["row_count"] = list(range(1, len(df) + 1))
        if "size" not in df.columns:
            df["size"] = 0
        if "retention_time" not in df.columns:
            df["retention_time"] = 0

        # Ensure all columns in types exist so astype(dtype=types) does not raise KeyError.
        for key in types.keys():
            if key not in df.columns:
                df[key] = pd.NA

        # If I understand corretcly, df.astype don't handle numeric numbers well.
        df["row_count"] = pd.to_numeric(df["row_count"])
        df["size"] = pd.to_numeric(df["size"])
        df["retention_time"] = pd.to_numeric(df["retention_time"])
        df = df.astype(dtype=types)
    

    # read_csv can do parse_dates, but most connectors don't have these fields so leaving it optional
    if "last_altered" in df:
        df["last_altered"] = pd.to_datetime(
            df["last_altered"], utc=True, format="mixed"
        )
        df["last_altered"] = df["last_altered"].apply(
            lambda x: (x.tz_convert(timezone.utc).replace(microsecond=0))
        )
    if "created" in df:
        df["created"] = pd.to_datetime(df["created"], utc=True, format="mixed")
        df["created"] = df["created"].apply(
            lambda x: (x.tz_convert(timezone.utc).replace(microsecond=0))
        )

    # Shkolar did it in playtika
    if "owner" in df:
        df = df.drop(columns=["owner"])

    return df


def load_columns(filepath_or_buffer, is_csv=True):
    types = {
        "database": "category",
        "schema": "category",
        "table_name": "category",
        "column_name": "string",
        "ordinal_position": "Int16",
        "data_type": "category",
        "default": "category",
        "is_nullable": "category",  # Convert to boolean?
        "length": "Int64",
        "scale": "Int16",
        "comment": "string",
    }

    if is_csv:
        df = pd.read_csv(
            filepath_or_buffer,
            dtype=types,
            encoding="ISO-8859-1",
        )
        for key in types.keys():
            if key not in df:
                df[key] = pd.NA
    else:
        if not isinstance(filepath_or_buffer, pd.DataFrame):
            df = pd.DataFrame(filepath_or_buffer)
        else:
            df = filepath_or_buffer.copy()

        # Happens for temp schema after reset
        if df.empty:
            return df

        # Ensure all columns in types exist so to_numeric/astype do not raise KeyError.
        for key in types.keys():
            if key not in df.columns:
                df[key] = pd.NA

        # If I understand corretcly, df.astype don't handle numeric numbers well.
        df["ordinal_position"] = pd.to_numeric(df["ordinal_position"])
        df["length"] = pd.to_numeric(df["length"])
        df["scale"] = pd.to_numeric(df["scale"])

        df = df.astype(dtype=types)

    return df


def load_views(filepath_or_buffer, is_csv=True):
    types = {
        "database": "category",
        "schema": "category",
        "name": "string",
        "view_definition": "string",
        "id": "string",
    }
    if is_csv:
        df = pd.read_csv(
            filepath_or_buffer,
            dtype=types,
        )
        for key in types.keys():
            if key not in df:
                df[key] = pd.NA
    else:
        if not isinstance(filepath_or_buffer, pd.DataFrame):
            df = pd.DataFrame(filepath_or_buffer)
        else:
            df = filepath_or_buffer.copy()
        if df.empty:
            return df
        for key in types.keys():
            if key not in df.columns:
                df[key] = pd.NA
        df = df.astype(dtype=types)

    return df


def load_queries(filepath_or_buffer):
    types = {
        "database": "category",
        "schema": "category",
        "end_time": "string",
        "query_text": "string",
        "count": "Int64",  # not sure Int16 is big enough (highest number - 32K)
        "id": "string",
    }

    # using chunks because of playtika. no idea why needed.
    iterator = pd.read_csv(
        filepath_or_buffer,
        keep_default_na=False,
        dtype=types,
        chunksize=100,
    )
    to_return = pd.concat(iterator, ignore_index=True)
    to_return["end_time"] = pd.to_datetime(
        to_return["end_time"], utc=True, format="mixed"
    )
    return to_return


def load_fks(filepath_or_buffer, is_csv=True):
    types = {
        "created_on": "string",
        "pk_database_name": "string",
        "pk_schema_name": "string",
        "pk_table_name": "string",
        "pk_column_name": "string",
        "fk_database_name": "string",
        "fk_schema_name": "string",
        "fk_table_name": "string",
        "fk_column_name": "string",
        "key_sequence": "string",
        "update_rule": "string",
        "delete_rule": "string",
        "fk_name": "string",
        "pk_name": "string",
        "deferrability": "string",
        "rely": "boolean",
    }
    if is_csv:
        df = pd.read_csv(
            filepath_or_buffer,
            dtype=types,
        )
        for key in types.keys():
            if key not in df:
                df[key] = pd.NA
    else:
        if not isinstance(filepath_or_buffer, pd.DataFrame):
            df = pd.DataFrame(filepath_or_buffer)
        else:
            df = filepath_or_buffer.copy()
        if df.empty:
            return df
        for key in types.keys():
            if key not in df.columns:
                df[key] = pd.NA
        df = df.astype(dtype=types)

    return df


def load_pks(filepath_or_buffer, is_csv=True):
    types = {
        "created_on": "string",
        "database_name": "string",
        "schema_name": "string",
        "table_name": "string",
        "column_name": "string",
        "key_sequence": "string",
        "constraint_name": "string",
        "rely": "string",
    }
    if is_csv:
        df = pd.read_csv(
            filepath_or_buffer,
            dtype=types,
        )
        for key in types.keys():
            if key not in df:
                df[key] = pd.NA
    else:
        if not isinstance(filepath_or_buffer, pd.DataFrame):
            df = pd.DataFrame(filepath_or_buffer)
        else:
            df = filepath_or_buffer.copy()
        if df.empty:
            return df
        for key in types.keys():
            if key not in df.columns:
                df[key] = pd.NA
        df = df.astype(dtype=types)

    return df


def load_terms(filepath, lines=True):
    if len(filepath) > 0:
        extension = os.path.splitext(filepath)[
            1
        ].lower()  # Extracts '.csv' or '.json' (lowercase)
        if extension == ".csv":
            terms_df = pd.read_csv(filepath)
        elif extension == ".json":
            terms_df = pd.read_json(filepath, lines=lines)
        else:
            raise Exception("Unknown extension.")

        return terms_df

    return None


def load_global_terms(filepath):
    global_terms = None
    if len(filepath) > 0:
        extension = os.path.splitext(filepath)[
            1
        ].lower()  # Extracts '.csv' or '.json' (lowercase)
        if extension == ".json":
            with open(filepath, "r") as file:
                global_terms = json.load(file)
            return global_terms
        else:
            raise Exception("Unknown extension.")

    return None
