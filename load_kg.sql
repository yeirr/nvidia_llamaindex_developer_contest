LOAD 'age';

SET search_path = ag_catalog, "$user", public;

-- Create new graph if missing.
SELECT * FROM create_graph('age_dev');

-- Create test two vertices and one edge.
SELECT * FROM cypher('age_dev', $$
    CREATE (:Gender {type: 'Human Male'})-[:REQUIRES]->(:Vitamins {type: 'Vitamin C'})
$$) AS (a agtype);

-- Showing the newly created tables.
SELECT * FROM ag_catalog.ag_label;
