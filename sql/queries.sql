
-- Query 1: Order counts by status
SELECT
    status,
    COUNT(*) AS order_count
FROM orders
GROUP BY status
ORDER BY status;


-- Query 2: Revenue by product category
SELECT
    p.category,
    ROUND(SUM(o.order_amount), 2) AS completed_revenue,
    COUNT(*) AS completed_orders
FROM orders AS o
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY completed_revenue DESC;


-- Query 3: Top 10 customers by completed-order spending
SELECT
    c.customer_id,
    c.city,
    COUNT(o.order_id) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS total_spending
FROM customers AS c
JOIN orders AS o
    ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.city
ORDER BY total_spending DESC
LIMIT 10;


-- Query 4: Average delivery time by product category
SELECT
    p.category,
    ROUND(AVG(o.delivery_minutes), 2) AS avg_delivery_minutes,
    COUNT(o.delivery_minutes) AS orders_with_delivery_data
FROM orders AS o
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY avg_delivery_minutes;


-- Query 5: Cancellation rate by city
SELECT
    c.city,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END)
        AS cancelled_orders,
    ROUND(
        100.0 * SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END)
        / COUNT(*),
        2
    ) AS cancellation_rate_pct
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
GROUP BY c.city
ORDER BY cancellation_rate_pct DESC;


-- Query 6: Monthly completed-order revenue
SELECT
    substr(o.order_ts, 1, 7) AS order_month,
    ROUND(SUM(o.order_amount), 2) AS monthly_revenue,
    COUNT(*) AS completed_orders
FROM orders AS o
WHERE o.status = 'completed'
GROUP BY substr(o.order_ts, 1, 7)
ORDER BY order_month;


-- Query 7: Customers with no completed orders
SELECT
    c.customer_id,
    c.city,
    c.age,
    c.gender
FROM customers AS c
WHERE NOT EXISTS (
    SELECT 1
    FROM orders AS o
    WHERE o.customer_id = c.customer_id
      AND o.status = 'completed'
)
ORDER BY c.customer_id;
