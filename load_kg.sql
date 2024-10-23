LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Create new graph if missing.
SELECT create_graph('age_dev');
