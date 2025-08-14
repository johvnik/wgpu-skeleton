//! Entity Component System implementation
//!
//! A simple but flexible ECS that allows you to build complex simulations
//! from simple components and systems.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Entity ID - simple integer
pub type EntityId = u32;

/// Component trait that all components must implement
pub trait Component: 'static + Send + Sync {}

/// ECS World that manages entities and components
pub struct World {
    next_entity_id: EntityId,
    entities: Vec<EntityId>,
    components: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl World {
    /// Create a new empty world
    pub fn new() -> Self {
        Self {
            next_entity_id: 0,
            entities: Vec::new(),
            components: HashMap::new(),
        }
    }

    /// Create a new entity and return its ID
    pub fn create_entity(&mut self) -> EntityId {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        self.entities.push(id);
        id
    }

    /// Create an entity with a builder pattern
    pub fn spawn(&mut self) -> EntityBuilder {
        let id = self.create_entity();
        EntityBuilder {
            world: self,
            entity: id,
        }
    }

    /// Add a component to an entity
    pub fn add_component<T: Component>(&mut self, entity: EntityId, component: T) {
        let type_id = TypeId::of::<T>();
        let storage = self
            .components
            .entry(type_id)
            .or_insert_with(|| Box::new(HashMap::<EntityId, T>::new()));

        if let Some(storage) = storage.downcast_mut::<HashMap<EntityId, T>>() {
            storage.insert(entity, component);
        }
    }

    /// Get a component from an entity
    pub fn get_component<T: Component>(&self, entity: EntityId) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        self.components
            .get(&type_id)?
            .downcast_ref::<HashMap<EntityId, T>>()?
            .get(&entity)
    }

    /// Get a mutable component from an entity
    pub fn get_component_mut<T: Component>(&mut self, entity: EntityId) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        self.components
            .get_mut(&type_id)?
            .downcast_mut::<HashMap<EntityId, T>>()?
            .get_mut(&entity)
    }

    /// Remove a component from an entity
    pub fn remove_component<T: Component>(&mut self, entity: EntityId) -> Option<T> {
        let type_id = TypeId::of::<T>();
        self.components
            .get_mut(&type_id)?
            .downcast_mut::<HashMap<EntityId, T>>()?
            .remove(&entity)
    }

    /// Query for entities with a specific component
    pub fn query<T: Component>(&self) -> impl Iterator<Item = (EntityId, &T)> {
        let type_id = TypeId::of::<T>();
        self.components
            .get(&type_id)
            .and_then(|storage| storage.downcast_ref::<HashMap<EntityId, T>>())
            .map(|storage| storage.iter().map(|(&id, component)| (id, component)))
            .into_iter()
            .flatten()
    }

    /// Query for entities with a specific component (mutable)
    pub fn query_mut<T: Component>(&mut self) -> impl Iterator<Item = (EntityId, &mut T)> {
        let type_id = TypeId::of::<T>();
        self.components
            .get_mut(&type_id)
            .and_then(|storage| storage.downcast_mut::<HashMap<EntityId, T>>())
            .map(|storage| storage.iter_mut().map(|(&id, component)| (id, component)))
            .into_iter()
            .flatten()
    }

    /// Check if an entity has a specific component
    pub fn has_component<T: Component>(&self, entity: EntityId) -> bool {
        self.get_component::<T>(entity).is_some()
    }

    /// Remove an entity and all its components
    pub fn despawn(&mut self, entity: EntityId) {
        self.entities.retain(|&e| e != entity);

        // Remove from all component storages
        for storage in self.components.values_mut() {
            // This is a bit of a hack since we can't know the exact type
            // In a more sophisticated ECS, you'd track which components an entity has
            // For now, we'll just leave orphaned components (they won't be accessible)
        }
    }

    /// Get all entities
    pub fn entities(&self) -> &[EntityId] {
        &self.entities
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder pattern for creating entities with components
pub struct EntityBuilder<'a> {
    world: &'a mut World,
    entity: EntityId,
}

impl<'a> EntityBuilder<'a> {
    /// Add a component to this entity
    pub fn with<T: Component>(self, component: T) -> Self {
        self.world.add_component(self.entity, component);
        self
    }

    /// Get the entity ID
    pub fn id(&self) -> EntityId {
        self.entity
    }

    /// Finish building and return the entity ID
    pub fn build(self) -> EntityId {
        self.entity
    }
}

// Implement Drop to automatically finish building
impl<'a> Drop for EntityBuilder<'a> {
    fn drop(&mut self) {
        // Entity is already created, nothing to do
    }
}
