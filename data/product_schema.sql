-- Demo e-commerce schema for the AI Data Agent.
--
-- Five tables: users, categories, products, orders, order_items.
-- Loaded automatically by docker-compose on first container startup
-- (mounted at /docker-entrypoint-initdb.d/01-schema.sql).
--
-- Convention notes:
--   * All monetary amounts are stored as DECIMAL(10,2) / (12,2) in RMB yuan
--     (matches the business knowledge we'll teach the LLM in Phase 5).
--   * Order `status`:
--         paid      — counted in 销售额 (sales)
--         refunded  — excluded from 销售额
--         cancelled — excluded from 销售额
--   * `created_at` defaults to CURRENT_TIMESTAMP for stable reproducibility.

CREATE DATABASE IF NOT EXISTS `demo` DEFAULT CHARACTER SET utf8mb4;
USE `demo`;

-- ---------------------------------------------------------------------------
-- users
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS `order_items`;
DROP TABLE IF EXISTS `orders`;
DROP TABLE IF EXISTS `products`;
DROP TABLE IF EXISTS `categories`;
DROP TABLE IF EXISTS `users`;

CREATE TABLE `users` (
  `id`         BIGINT       NOT NULL AUTO_INCREMENT,
  `username`   VARCHAR(64)  NOT NULL,
  `email`      VARCHAR(128) NOT NULL,
  `city`       VARCHAR(64)  NOT NULL DEFAULT '未知',
  `created_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_users_username` (`username`),
  KEY         `idx_users_city`     (`city`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- categories
-- ---------------------------------------------------------------------------
CREATE TABLE `categories` (
  `id`          BIGINT       NOT NULL AUTO_INCREMENT,
  `name`        VARCHAR(64)  NOT NULL,
  `description` VARCHAR(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_categories_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- products
-- ---------------------------------------------------------------------------
CREATE TABLE `products` (
  `id`          BIGINT         NOT NULL AUTO_INCREMENT,
  `name`        VARCHAR(128)   NOT NULL,
  `category_id` BIGINT         NOT NULL,
  `price`       DECIMAL(10, 2) NOT NULL,
  `stock`       INT            NOT NULL DEFAULT 0,
  `status`      VARCHAR(16)    NOT NULL DEFAULT 'on_sale',
  `created_at`  DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_products_category` (`category_id`),
  CONSTRAINT `fk_products_category`
      FOREIGN KEY (`category_id`) REFERENCES `categories` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- orders
-- ---------------------------------------------------------------------------
CREATE TABLE `orders` (
  `id`           BIGINT         NOT NULL AUTO_INCREMENT,
  `user_id`      BIGINT         NOT NULL,
  `total_amount` DECIMAL(12, 2) NOT NULL,
  `status`       VARCHAR(16)    NOT NULL DEFAULT 'paid',
  `created_at`   DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_orders_user`        (`user_id`),
  KEY `idx_orders_created_at`  (`created_at`),
  KEY `idx_orders_status`      (`status`),
  CONSTRAINT `fk_orders_user`
      FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- order_items
-- ---------------------------------------------------------------------------
CREATE TABLE `order_items` (
  `id`         BIGINT         NOT NULL AUTO_INCREMENT,
  `order_id`   BIGINT         NOT NULL,
  `product_id` BIGINT         NOT NULL,
  `quantity`   INT            NOT NULL,
  `unit_price` DECIMAL(10, 2) NOT NULL,
  `amount`     DECIMAL(12, 2) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_oi_order`   (`order_id`),
  KEY `idx_oi_product` (`product_id`),
  CONSTRAINT `fk_oi_order`
      FOREIGN KEY (`order_id`) REFERENCES `orders` (`id`),
  CONSTRAINT `fk_oi_product`
      FOREIGN KEY (`product_id`) REFERENCES `products` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
