from sql2nl_redshift.heuristics import explain_sql

def test_basic_select():
    sql = "SELECT * FROM public.users LIMIT 5;"
    out = explain_sql(sql).lower()
    assert "selects all columns" in out
    assert "limits output to 5" in out

def test_group_by():
    sql = "SELECT user_id, COUNT(*) FROM public.orders GROUP BY 1 ORDER BY 2 DESC;"
    out = explain_sql(sql).lower()
    assert "aggregates using group by" in out
    assert "orders the result" in out
