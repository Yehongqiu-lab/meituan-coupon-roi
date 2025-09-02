import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def reduce_mem_usage(df, verbose=True):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        
        if pd.api.types.is_integer_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype("Int8") # use pandas nullable integer dtype
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype("Int16")
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype("Int32")
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype("Int64")  
        elif pd.api.types.is_float_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype("float32")
            else:
                df[col] = df[col].astype("float64")    
                    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def id_reassign_with_map(cols, *dfs, verbose=True, suffix="_code"):
    """Factorize IDs across multiple dfs; return new dfs + {col: mapping}."""
    mappings = {}
    new_dfs = list(dfs)

    for col in cols:
        merged = pd.concat([df[col].astype("string") for df in new_dfs], ignore_index=True)
        codes, uniques = pd.factorize(merged)  # -1 in codes means missing
        mapping = {val: int(i) for i, val in enumerate(uniques)}
        mappings[col] = mapping

        # memory report
        if verbose:
            before = merged.memory_usage(deep=True) / 1024**2
            after = codes.nbytes / 1024**2
            print(f"[{col}] ↓ to {after:5.2f} MB ({100*(before-after)/before:.1f}% reduction)")

        # write back
        offset = 0
        for i, df in enumerate(new_dfs):
            n = len(df)
            code_col = f"{col}{suffix}"
            df[code_col] = codes[offset:offset+n]
            # pick tight int dtype
            maxv = int(df[code_col].max())
            if maxv < 128:
                df[code_col] = df[code_col].astype("Int8")
            elif maxv < 32768:
                df[code_col] = df[code_col].astype("Int16")
            else:
                df[code_col] = df[code_col].astype("Int32")
            offset += n

    return tuple(new_dfs), mappings

def drop_original_id(cols, *dfs, suffix="_code"):
    """Drop original ID columns."""
    new_dfs = []
    for df in dfs:
        df = df.copy()
        for col in cols:
            code_col = f"{col}{suffix}"
            if code_col in df.columns:
                df = df.drop(columns=[col])
        new_dfs.append(df)
    return tuple(new_dfs)

def drop_duplicate_rows(*dfs):
    """Drop duplicate rows in each df."""
    new_dfs = []
    for df in dfs:
        n_before = len(df)
        df = df.drop_duplicates()
        n_after = len(df)
        if n_before != n_after:
            print(f"Dropped {n_before - n_after} duplicate rows")
        new_dfs.append(df)
    return tuple(new_dfs)

